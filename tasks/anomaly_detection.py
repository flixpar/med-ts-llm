import torch
import torch.nn.functional as F

from tqdm import tqdm

from .base import BaseTask


class AnomalyDetectionTask(BaseTask):

    def __init__(self, run_id, config):
        self.task = "anomaly_detection"
        assert config.history_len == config.pred_len, "Anomaly detection task requires history_len == pred_len"
        super(AnomalyDetectionTask, self).__init__(run_id, config)

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
        self.log_end()

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
        return {}

    def score(self, pred, target):
        return {
            "recon_mse": F.mse_loss(pred, target),
            "recon_mae": F.l1_loss(pred, target),
        }
