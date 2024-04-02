import torch
import torch.nn.functional as F

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

        self.model.eval()

    def val(self):
        preds, targets = self.predict(self.val_dataloader)
        scores = self.score(preds, targets)
        scores = {f"val/{metric}": value for metric, value in scores.items()}
        self.log_scores(scores)
        return scores

    def test(self):
        preds, targets = self.predict(self.test_dataloader)
        scores = self.score(preds, targets)
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
                    time_idx = inds[0] if isinstance(inds, tuple) else inds
                    time_idx = slice(time_idx, time_idx + pred.size(1))
                    cls_idx = 1 if n_classes == 2 else slice(None)

                    preds[time_idx, cls_idx] = pred[j].squeeze().cpu().detach()
                    targets[time_idx] = inputs["labels"][j].squeeze().cpu().detach()

        if n_classes == 2:
            preds[:, 0] = 1 - preds[:, 1]

        if step_size > pred_len:
            cutoff = n_points - (n_points % step_size)
            preds, targets = preds[:cutoff,:], targets[:cutoff]
            preds = preds.reshape(-1, step_size, n_classes)[:, :pred_len, :].reshape(-1, n_classes)
            targets = targets.reshape(-1, step_size)[:, :pred_len].reshape(-1)

        assert not torch.isnan(preds).any()
        assert not (targets < 0).any()

        return preds, targets

    def build_loss(self):
            is_binary = (self.train_dataset.n_classes == 2)
            match self.config.training.loss, is_binary:
                case "bce", True:
                    self.loss_fn = torch.nn.BCEWithLogitsLoss()
                case ("ce", False) | ("cross_entropy", False):
                    self.loss_fn = torch.nn.CrossEntropyLoss()
                case ("ce", True) | ("cross_entropy", True):
                    self.loss_fn = torch.nn.BCEWithLogitsLoss()
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
