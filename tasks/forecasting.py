import torch
import torch.nn.functional as F

from tqdm import tqdm

from .base import BaseTask


class ForecastTask(BaseTask):

    def __init__(self, run_id, config, newrun=True):
        self.task = "forecasting"
        super(ForecastTask, self).__init__(run_id, config, newrun)

    def train(self):
        for epoch in range(self.config.training.epochs):
            print(f"Epoch {epoch + 1}/{self.config.training.epochs}")
            self.model.train()
            for inputs in tqdm(self.train_dataloader):
                inputs = self.prepare_batch(inputs)

                pred = self.model(inputs)
                loss = self.loss_fn(pred, inputs["y"])

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
                batch_scores = self.score(pred, inputs["y"])
                scores.append(batch_scores)

        mean_scores = {f"val/{metric}": sum([s[metric] for s in scores]) / len(scores) for metric in scores[0].keys()}
        return mean_scores

    def test(self):
        self.model.eval()
        scores = []
        with torch.no_grad():
            for inputs in tqdm(self.test_dataloader):
                inputs = self.prepare_batch(inputs)

                pred = self.model(inputs)
                batch_scores = self.score(pred, inputs["y"])
                scores.append(batch_scores)

        mean_scores = {f"test/{metric}": sum([s[metric] for s in scores]) / len(scores) for metric in scores[0].keys()}
        return mean_scores

    def score(self, pred, target):
        return {
            "mse": F.mse_loss(pred, target).item(),
            "mae": F.l1_loss(pred, target).item(),
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
