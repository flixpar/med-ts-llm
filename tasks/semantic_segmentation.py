import torch
import torch.nn.functional as F

import pytorch_optimizer

import plotly.graph_objects as go

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    jaccard_score,
)

from tqdm import tqdm

from .base import BaseTask


class SemanticSegmentationTask(BaseTask):

    def __init__(self, run_id, config, newrun=True):
        self.task = "semantic_segmentation"
        super(SemanticSegmentationTask, self).__init__(run_id, config, newrun)

    def train(self):
        for epoch in range(self.config.training.epochs):
            print(f"Epoch {epoch + 1}/{self.config.training.epochs}")
            self.model.train()
            for inputs in tqdm(self.train_dataloader):
                inputs = self.prepare_batch(inputs)

                with torch.autocast(self.device.type, dtype=torch.bfloat16, enabled=self.mixed):
                    pred = self.model(inputs)
                    if pred.ndim == 3:
                        pred = pred.permute(0, 2, 1)
                        labels = inputs["labels"]
                    else:
                        labels = inputs["labels"].to(self.dtype)
                    loss = self.loss_fn(pred, labels)

                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()

                self.log_step(loss.item())

            val_scores = self.val()
            self.log_epoch(val_scores)
            self.scheduler.step()

        self.model.eval()

    def val(self):
        preds, targets = self.predict(self.val_dataloader)

        scores = self.score(preds, targets)
        scores = {f"val/{metric}": value for metric, value in scores.items()}
        self.log_scores(scores)

        preds_fig = self.plot_predictions(preds, targets)
        self.logger.log_figure(preds_fig, "val/predictions")

        return scores

    def test(self):
        preds, targets = self.predict(self.test_dataloader)

        scores = self.score(preds, targets)
        scores = {f"test/{metric}": value for metric, value in scores.items()}
        self.log_scores(scores)

        preds_fig = self.plot_predictions(preds, targets)
        self.logger.log_figure(preds_fig, "test/predictions")

        return scores

    def predict(self, dataloader):
        self.model.eval()

        dataset = dataloader.dataset
        pred_len = self.config.pred_len
        step_size = dataset.step_size
        n_points = dataset.n_points if dataset.clip_dataset else pred_len + ((len(dataset) - 1) * step_size)

        n_classes = dataset.n_classes
        bs = dataloader.batch_size

        preds = torch.full((n_points, n_classes), float("nan"))
        targets = torch.full((n_points,), -1, dtype=torch.int)

        with torch.no_grad():
            for idx, inputs in tqdm(enumerate(dataloader), total=len(dataloader)):
                inputs = self.prepare_batch(inputs)
                pred = self.model(inputs)

                for j in range(pred.size(0)):
                    inds = dataset.inverse_index((idx * bs) + j)
                    time_inds = slice(*(inds[0] if dataset.univariate else inds))
                    cls_idx = 1 if n_classes == 2 else slice(None)

                    preds[time_inds, cls_idx] = pred[j].squeeze().cpu().detach()
                    targets[time_inds] = inputs["labels"][j].squeeze().cpu().detach()

        if n_classes == 2:
            preds[:, 0] = 1 - preds[:, 1]

        if dataset.clip_dataset:
            mask = dataset.mask
            preds, targets = preds[mask,:], targets[mask]
        elif step_size > pred_len:
            cutoff = n_points - (n_points % step_size)
            preds, targets = preds[:cutoff,:], targets[:cutoff]
            preds = preds.reshape(-1, step_size, n_classes)[:, :pred_len, :].reshape(-1, n_classes)
            targets = targets.reshape(-1, step_size)[:, :pred_len].reshape(-1)

        assert not preds.isnan().any()
        assert not (targets < 0).any()

        return preds, targets

    def build_loss(self):
        is_binary = (self.train_dataset.n_classes == 2)
        match self.config.training.loss, is_binary:
            case ("bce" | "ce" | "cross_entropy" | "auto"), True:
                self.loss_fn = torch.nn.BCEWithLogitsLoss()
            case ("ce" | "cross_entropy" | "auto"), False:
                self.loss_fn = torch.nn.CrossEntropyLoss()
            case ("iou" | "jaccard"), b:
                self.loss_fn = pytorch_optimizer.JaccardLoss("binary" if b else "multiclass")
            case ("lovasz" | "lovasz-hinge"), True:
                self.loss_fn = pytorch_optimizer.LovaszHingeLoss(False)
            case _:
                raise ValueError(f"Invalid loss function selection: {self.config.training.loss}")
        return self.loss_fn

    def score(self, pred_scores, target):
        avg_mode = "binary" if pred_scores.size(1) == 2 else "macro"
        pred = pred_scores.argmax(dim=1).int().numpy()
        target = target.numpy()
        return {
            "accuracy": accuracy_score(target, pred),
            "f1": f1_score(target, pred, average=avg_mode, zero_division=0),
            "precision": precision_score(target, pred, average=avg_mode, zero_division=0),
            "recall": recall_score(target, pred, average=avg_mode, zero_division=0),
            "iou": jaccard_score(target, pred, average=avg_mode, zero_division=0),
        }

    def plot_predictions(self, pred_scores, targets, xrange=(0, 1000)):
        xinds = slice(*xrange)
        preds = pred_scores.argmax(dim=1).int()

        if pred_scores.size(1) == 2:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=torch.arange(*xrange), y=targets[xinds], mode="lines", name="target"))
            fig.add_trace(go.Scatter(x=torch.arange(*xrange), y=pred_scores[xinds, 1], mode="lines", name="pred"))
        else:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=torch.arange(*xrange), y=targets[xinds], mode="lines", name="target"))
            fig.add_trace(go.Scatter(x=torch.arange(*xrange), y=preds[xinds], mode="lines", name="pred"))

        return fig
