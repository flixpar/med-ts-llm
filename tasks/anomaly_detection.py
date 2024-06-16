import torch
import torch.nn.functional as F

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    jaccard_score,
    roc_auc_score,
)

from numba import jit
from bayes_opt import BayesianOptimization

from tqdm import tqdm

from .base import BaseTask
from utils import dict_to_object


class AnomalyDetectionTask(BaseTask):

    def __init__(self, run_id, config, newrun=True):
        self.task = "anomaly_detection"
        self.task_config = config.tasks.anomaly_detection
        assert config.history_len == config.pred_len, "Anomaly detection task requires history_len == pred_len"
        assert self.task_config.get("score_metric", "mse") == "mse"
        super(AnomalyDetectionTask, self).__init__(run_id, config, newrun)

    def train(self):
        for epoch in range(self.config.training.epochs):
            print(f"Epoch {epoch + 1}/{self.config.training.epochs}")
            self.model.train()
            for inputs in tqdm(self.train_dataloader):
                inputs = self.prepare_batch(inputs)

                with torch.autocast(self.device.type, dtype=torch.bfloat16, enabled=self.mixed):
                    pred = self.model(inputs)
                    loss = self.loss_fn(pred, inputs["x_enc"].detach())

                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()

                self.log_step(loss.item())

            val_scores = self.val()
            self.log_epoch(val_scores)
            self.scheduler.step()

        self.model.eval()

    def val(self):
        results = self.predict(self.val_dataloader, split="val")

        anom_scores = self.score_anomalies(results.anomaly_preds, results.anomaly_labels)
        recon_scores = self.score(results.recon_preds, results.recon_targets)
        thresholds = {"anomaly_quantile": results.anomaly_quantile, "anomaly_threshold": results.anomaly_threshold}
        scores = anom_scores | recon_scores | thresholds
        scores = {f"val/{metric}": value for metric, value in scores.items()}
        self.log_scores(scores)

        return scores

    def test(self):
        results = self.predict(self.test_dataloader, split="test")

        anom_scores = self.score_anomalies(results.anomaly_preds, results.anomaly_labels)
        recon_scores = self.score(results.recon_preds, results.recon_targets)
        thresholds = {"anomaly_quantile": results.anomaly_quantile, "anomaly_threshold": results.anomaly_threshold}
        scores = anom_scores | recon_scores | thresholds
        scores = {f"test/{metric}": value for metric, value in scores.items()}
        self.log_scores(scores)

        return scores

    def predict(self, dataloader, split=None):
        self.model.eval()

        dataset = dataloader.dataset
        pred_len = self.config.pred_len
        step_size = dataset.step_size
        n_points = dataset.n_points if dataset.clip_dataset else pred_len + ((len(dataset) - 1) * step_size)

        n_features = dataset.real_features
        bs = dataloader.batch_size

        preds = torch.full((n_points, n_features), float("nan"))
        targets = torch.full((n_points, n_features), float("nan"))
        labels = torch.full((n_points,), -1, dtype=torch.int)

        with torch.no_grad():
            for idx, inputs in tqdm(enumerate(dataloader), total=len(dataloader)):
                inputs = self.prepare_batch(inputs)
                pred = self.model(inputs)

                for j in range(pred.size(0)):
                    inds = dataset.inverse_index((idx * bs) + j)
                    time_inds, feature_inds = inds if dataset.univariate else (inds, slice(None))
                    time_inds = slice(*time_inds)

                    preds[time_inds, feature_inds] = pred[j].squeeze().cpu().detach()
                    targets[time_inds, feature_inds] = inputs["x_enc"][j].squeeze().cpu().detach()
                    labels[time_inds] = inputs["labels"][j].squeeze().cpu().detach()

        if dataset.clip_dataset:
            mask = dataset.mask
            preds, targets, labels = preds[mask,:], targets[mask,:], labels[mask]
        elif step_size > pred_len:
            cutoff = n_points - (n_points % step_size)
            preds, targets, labels = preds[:cutoff,:], targets[:cutoff,:], labels[:cutoff]
            preds = preds.reshape(-1, step_size, n_features)[:, :pred_len, :].reshape(-1, n_features)
            targets = targets.reshape(-1, step_size, n_features)[:, :pred_len, :].reshape(-1, n_features)
            labels = labels.reshape(-1, step_size)[:, :pred_len].reshape(-1)

        assert not preds.isnan().any()
        assert not targets.isnan().any()
        assert not (labels < 0).any()

        scores = F.mse_loss(preds, targets, reduction="none")
        if self.task_config.normalize_by_feature:
            mean_scores = scores.mean(dim=0)
            scores = scores / mean_scores.unsqueeze(0)
        scores = scores.nanmean(dim=1)
        if (window := self.task_config.get("normalize_moving_window", 0)) > 0:
            moving_avg = running_mean(scores, window)
            scores = scores / moving_avg

        match self.task_config.threshold, split:
            case "optimize", _:
                quantile = optimize_threshold(scores, labels)
            case "optimize-test", "test":
                quantile = optimize_threshold(scores, labels)
            case ("auto" | "optimize-test"), _:
                quantile = 1 - (labels.sum().item() / (n_points + self.train_dataset.n_points))
            case float(), _:
                quantile = 1 - self.task_config.threshold
            case _:
                raise ValueError(f"Invalid threshold selection: {self.task_config.threshold}")

        threshold = scores.quantile(quantile)
        anomalies = (scores > threshold).to(torch.int)
        anomalies = adjust_anomalies(anomalies, labels)

        results = {
            "recon_preds": preds,
            "recon_targets": targets,
            "anomaly_labels": labels,
            "anomaly_scores": scores,
            "anomaly_preds": anomalies,
            "anomaly_quantile": quantile,
            "anomaly_threshold": threshold.item(),
        }
        return dict_to_object(results)

    def score(self, pred, target):
        return {
            "recon_mse": F.mse_loss(pred, target).item(),
            "recon_mae": F.l1_loss(pred, target).item(),
        }

    def score_anomalies(self, pred, target):
        pred, target = pred.cpu().numpy(), target.cpu().numpy()
        return {
            "accuracy": accuracy_score(target, pred),
            "f1": f1_score(target, pred, average="binary", zero_division=0),
            "auroc": roc_auc_score(target, pred),
            "precision": precision_score(target, pred, average="binary", zero_division=0),
            "recall": recall_score(target, pred, average="binary", zero_division=0),
            "iou": jaccard_score(target, pred, average="binary", zero_division=0),
        }

    def build_loss(self):
        match self.config.training.loss:
            case "mse":
                self.loss_fn = torch.nn.MSELoss()
            case "mae":
                self.loss_fn = torch.nn.L1Loss()
            case "smooth_l1" | "smooth_mae":
                self.loss_fn = torch.nn.SmoothL1Loss()
            case _:
                raise ValueError(f"Invalid loss function selection: {self.config.training.loss}")
        return self.loss_fn


def adjust_anomalies(pred, gt):
    if use_torch := isinstance(pred, torch.Tensor):
        pred, gt = pred.numpy(), gt.numpy()
    pred = _adjust_anomalies(pred, gt)
    if use_torch:
        pred = torch.tensor(pred, dtype=torch.int)
    return pred

@jit(nopython=True)
def _adjust_anomalies(pred, gt):
    anomaly_state = False
    for i in range(len(gt)):
        if gt[i] == 1 and pred[i] == 1 and not anomaly_state:
            anomaly_state = True
            for j in range(i, 0, -1):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
            for j in range(i, len(gt)):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
        elif gt[i] == 0:
            anomaly_state = False
        if anomaly_state:
            pred[i] = 1
    return pred

def running_mean(xs, window_size):
    if window_size % 2 == 0:
        window_size += 1
    kernel = torch.ones(1, 1, window_size) / window_size
    xs = F.conv1d(xs.view(1, 1, -1), kernel, stride=1, padding="same").squeeze()
    return xs

def optimize_threshold(scores, labels):
    def score_func(q):
        threshold = scores.quantile(q)
        anomalies = (scores > threshold).to(torch.int)
        anomalies = adjust_anomalies(anomalies, labels)
        return f1_score(labels, anomalies, average="binary", zero_division=0)

    optimizer = BayesianOptimization(
        f = score_func,
        pbounds = {"q": (0.5, 1.0)},
        random_state = 0,
        verbose = 0,
    )
    optimizer.maximize(init_points=10, n_iter=20)
    return optimizer.max["params"]["q"]
