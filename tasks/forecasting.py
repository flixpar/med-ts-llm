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

                with torch.autocast(self.device.type, dtype=torch.bfloat16, enabled=self.mixed):
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
        preds, targets = self.predict(self.val_dataloader)
        scores = self.score(preds, targets)
        scores = {f"val/{k}": v for k, v in scores.items()}
        self.log_scores(scores)
        return scores

    def test(self):
        preds, targets = self.predict(self.test_dataloader)
        scores = self.score(preds, targets)
        scores = {f"test/{k}": v for k, v in scores.items()}
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

        with torch.no_grad():
            for idx, inputs in tqdm(enumerate(dataloader), total=len(dataloader)):
                inputs = self.prepare_batch(inputs)
                pred = self.model(inputs)

                for j in range(pred.size(0)):
                    inds = dataset.inverse_index((idx * bs) + j)
                    time_idx, feature_idx = inds if isinstance(inds, tuple) else (inds, slice(None))
                    time_idx = slice(time_idx, time_idx + pred.size(1))

                    preds[time_idx, feature_idx] = pred[j].squeeze().cpu().detach()
                    targets[time_idx, feature_idx] = inputs["y"][j].squeeze().cpu().detach()

        if step_size > pred_len:
            cutoff = n_points - (n_points % step_size)
            preds, targets = preds[:cutoff,:], targets[:cutoff,:]
            preds = preds.reshape(-1, step_size, n_features)[:, :pred_len, :].reshape(-1, n_features)
            targets = targets.reshape(-1, step_size, n_features)[:, :pred_len, :].reshape(-1, n_features)

        assert not torch.isnan(preds).any()
        assert not torch.isnan(targets).any()

        return preds, targets

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
