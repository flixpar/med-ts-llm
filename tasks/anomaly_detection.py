import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from tqdm import tqdm

from .base import BaseTask


class AnomalyDetectionTask(BaseTask):

    def __init__(self, run_id, config, newrun=True):
        self.task = "anomaly_detection"
        assert config.history_len == config.pred_len, "Anomaly detection task requires history_len == pred_len"
        super(AnomalyDetectionTask, self).__init__(run_id, config, newrun)

    def train(self):
        for epoch in range(self.config.training.epochs):
            print(f"Epoch {epoch + 1}/{self.config.training.epochs}")
            self.model.train()
            for (x_enc, x_dec, _) in tqdm(self.train_dataloader):
                x_enc = x_enc.to(self.device, self.dtype)
                x_dec = x_dec.to(self.device, self.dtype)

                pred = self.model(x_enc, x_dec)
                loss = self.loss_fn(pred, x_enc.detach())

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
            for (x_enc, x_dec, label) in tqdm(self.val_dataloader):
                x_enc = x_enc.to(self.device, self.dtype)
                x_dec = x_dec.to(self.device, self.dtype)
                label = label.to(self.device, self.dtype)

                pred = self.model(x_enc, x_dec)
                batch_scores = self.score(pred, x_enc.detach())
                scores.append(batch_scores)

        mean_scores = {f"val_{metric}": sum([s[metric] for s in scores]) / len(scores) for metric in scores[0].keys()}
        return mean_scores

    def test(self):
        self.model.eval()

        self.test_dataloader = DataLoader(
            self.test_dataset,
            batch_size = 1,
            shuffle = False,
            num_workers = self.config.setup.num_workers,
        )

        assert self.test_dataset.step_size == self.config.pred_len
        assert self.test_dataloader.batch_size == 1

        dataset_len = self.test_dataset.n_points - (self.test_dataset.n_points % self.config.pred_len)
        is_multi2uni = self.test_dataset.__class__.__name__ == "Multi2UniDataset"
        n_features = self.test_dataset.real_features if is_multi2uni else self.test_dataset.n_features

        preds = torch.full((dataset_len, n_features), float("nan"))
        targets = torch.full((dataset_len, n_features), float("nan"))
        labels = torch.full((dataset_len,), -1, dtype=torch.int)

        with torch.no_grad():
            for idx, (x_enc, x_dec, label) in tqdm(enumerate(self.test_dataloader), total=len(self.test_dataloader)):
                x_enc = x_enc.to(self.device, self.dtype)
                x_dec = x_dec.to(self.device, self.dtype)
                label = label.to(self.device, self.dtype)

                pred = self.model(x_enc, x_dec)

                pred, x_enc, label = pred.squeeze(), x_enc.squeeze(), label.squeeze()

                if is_multi2uni:
                    time_idx, feature_idx = self.test_dataset.inverse_index(idx)
                else:
                    time_idx, feature_idx = idx, slice(None)
                time_idx = slice(time_idx, time_idx + pred.size(0))

                preds[time_idx, feature_idx] = pred.cpu().detach()
                targets[time_idx, feature_idx] = x_enc.cpu().detach()
                labels[time_idx] = label.cpu().detach()

        assert not torch.isnan(preds).any()
        assert not torch.isnan(targets).any()
        assert not (labels < 0).any()

        scores = F.mse_loss(preds, targets, reduction="none").mean(dim=-1)
        threshold = scores.quantile(torch.tensor(0.9))
        anomalies = (scores > threshold).to(torch.int)

        anomalies = adjust_anomalies(anomalies, labels)

        scores = self.score_anomalies(anomalies, labels)
        scores = {f"test_{metric}": value for metric, value in scores.items()}
        self.log_scores(scores)

        return scores

    def score(self, pred, target):
        return {
            "recon_mse": F.mse_loss(pred, target).item(),
            "recon_mae": F.l1_loss(pred, target).item(),
        }

    def score_anomalies(self, pred, target):
        accuracy = (pred == target).float().mean()
        precision = (pred * target).sum() / pred.sum()
        recall = (pred * target).sum() / target.sum()
        f1 = 2 * (precision * recall) / (precision + recall)
        return {
            "accuracy": accuracy.item(),
            "precision": precision.item(),
            "recall": recall.item(),
            "f1": f1.item(),
        }

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
