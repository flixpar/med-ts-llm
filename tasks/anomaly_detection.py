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

from tqdm import tqdm

from .base import BaseTask
from utils import dict_to_object


class AnomalyDetectionTask(BaseTask):

    def __init__(self, run_id, config, newrun=True):
        self.task = "anomaly_detection"
        assert config.history_len == config.pred_len, "Anomaly detection task requires history_len == pred_len"
        super(AnomalyDetectionTask, self).__init__(run_id, config, newrun)

    def train(self):
        for epoch in range(self.config.training.epochs):
            print(f"Epoch {epoch + 1}/{self.config.training.epochs}")
            self.model.train()
            for inputs in tqdm(self.train_dataloader):
                inputs = self.prepare_batch(inputs)

                pred = self.model(inputs)
                loss = self.loss_fn(pred, inputs["x_enc"].detach())

                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()

                self.log_step(loss.item())

            val_scores = self.val()
            self.log_epoch(val_scores)

        self.model.eval()

    def val(self):
        self.model.eval()
        scores = []
        with torch.no_grad():
            for inputs in tqdm(self.val_dataloader):
                inputs = self.prepare_batch(inputs)

                pred = self.model(inputs)
                batch_scores = self.score(pred, inputs["x_enc"].detach())
                scores.append(batch_scores)

        mean_scores = {f"val/{metric}": sum([s[metric] for s in scores]) / len(scores) for metric in scores[0].keys()}
        return mean_scores

    def test(self):
        results = self.predict(self.test_dataloader)

        scores = self.score_anomalies(results.anomaly_preds, results.anomaly_labels)
        scores = {f"test/{metric}": value for metric, value in scores.items()}
        self.log_scores(scores)

        return scores

    def predict(self, dataloader):
        self.model.eval()

        dataset = dataloader.dataset
        pred_len = self.config.pred_len
        step_size = dataset.step_size
        dataset_len = ((dataset.n_points - pred_len) // step_size) + 1
        n_points = pred_len + ((dataset_len - 1) * step_size)

        n_features = getattr(dataset, "real_features", dataset.n_features)
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
                    time_idx, feature_idx = inds if isinstance(inds, tuple) else (inds, slice(None))
                    time_idx = slice(time_idx, time_idx + pred.size(1))

                    preds[time_idx, feature_idx] = pred[j].squeeze().cpu().detach()
                    targets[time_idx, feature_idx] = inputs["x_enc"][j].squeeze().cpu().detach()
                    labels[time_idx] = inputs["labels"][j].squeeze().cpu().detach()

        assert not torch.isnan(preds).any()
        assert not torch.isnan(targets).any()
        assert not (labels < 0).any()

        scores = F.mse_loss(preds, targets, reduction="none")
        mean_scores = scores.mean(dim=0)
        scores = scores / mean_scores.unsqueeze(0)
        scores = scores.nanmean(dim=1)

        threshold = scores.quantile(torch.tensor(0.9))
        anomalies = (scores > threshold).to(torch.int)
        anomalies = adjust_anomalies(anomalies, labels)

        results = {
            "recon_preds": preds,
            "recon_targets": targets,
            "anomaly_labels": labels,
            "anomaly_scores": scores,
            "anomaly_preds": anomalies
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
    pred = pred.clone()
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
