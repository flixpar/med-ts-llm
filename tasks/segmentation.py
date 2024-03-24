import torch
import torch.nn.functional as F

import math
import scipy

from tqdm import tqdm

from .base import BaseTask


class SegmentationTask(BaseTask):

    def __init__(self, run_id, config, newrun=True):
        self.task = "segmentation"
        super(SegmentationTask, self).__init__(run_id, config, newrun)

        self.segmentation_mode = self.config.tasks.segmentation.mode
        assert self.segmentation_mode == "boundary-prediction"

    def train(self):
        for epoch in range(self.config.training.epochs):
            print(f"Epoch {epoch + 1}/{self.config.training.epochs}")
            self.model.train()
            for inputs in tqdm(self.train_dataloader):
                inputs = self.prepare_batch(inputs)

                pred = self.model(inputs)
                loss = self.loss_fn(pred, inputs["labels"])

                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()

                self.log_step(loss.item())

            val_scores = self.val()
            self.log_epoch(val_scores)

        self.model.eval()

    def val(self):
        results = self.predict(self.val_dataloader)
        scores = self.score(results)
        scores = {f"val/{metric}": value for metric, value in scores.items()}
        self.log_scores(scores)
        return scores

    def test(self):
        results = self.predict(self.test_dataloader)
        scores = self.score(results)
        scores = {f"test/{metric}": value for metric, value in scores.items()}
        self.log_scores(scores)
        return scores

    def build_loss(self):
        match self.config.training.loss:
            case "bce":
                self.loss_fn = torch.nn.BCEWithLogitsLoss()
            case _:
                raise ValueError(f"Invalid loss function selection: {self.config.training.loss}")
        return self.loss_fn

    def predict(self, dataloader):
        self.model.eval()

        dataset = dataloader.dataset
        pred_len = self.config.pred_len
        step_size = dataset.step_size
        dataset_len = ((dataset.n_points - pred_len) // step_size) + 1
        n_points = pred_len + ((dataset_len - 1) * step_size)
        bs = dataloader.batch_size

        preds = torch.full((n_points,), float("nan"))
        targets = torch.full((n_points,), -1, dtype=torch.int)

        with torch.no_grad():
            for idx, inputs in tqdm(enumerate(dataloader), total=len(dataloader)):
                inputs = self.prepare_batch(inputs)
                pred = self.model(inputs)

                for j in range(pred.size(0)):
                    inds = dataset.inverse_index((idx * bs) + j)
                    time_idx = inds[0] if isinstance(inds, tuple) else inds
                    time_idx = slice(time_idx, time_idx + pred.size(1))

                    preds[time_idx] = pred[j].squeeze().cpu().detach()
                    targets[time_idx] = inputs["labels"][j].squeeze().cpu().detach()

        assert not torch.isnan(preds).any()
        assert not (targets < 0).any()

        pred_scores = preds.clone()

        # pred_scores = smooth_scores(pred_scores, 25, "mean")
        # pred_points = find_peaks_threshold(pred_scores, 0.55)

        distance_thresh = self.config.tasks.segmentation.distance_thresh
        pred_points = scipy.signal.find_peaks(pred_scores.numpy(), distance=distance_thresh)[0]
        pred_points = torch.tensor(pred_points, dtype=torch.int)

        pred_labels = torch.zeros_like(targets)
        pred_labels[pred_points] = 1

        label_points = targets.nonzero().squeeze()

        pred_segments = torch.cat([torch.tensor([0]), pred_points, torch.tensor([len(pred_scores)-1])])
        pred_segments = pred_segments.unfold(0, 2, 1)

        label_segments = torch.cat([torch.tensor([0]), label_points, torch.tensor([len(pred_scores)-1])])
        label_segments = label_segments.unfold(0, 2, 1)

        return {
            "preds_raw": preds,
            "pred_points": pred_points,
            "pred_labels": pred_labels,
            "pred_segments": pred_segments,
            "labels": targets,
            "label_points": label_points,
            "label_segments": label_segments,
        }

    def score(self, results):
        pred_labels = results["pred_labels"]
        target_labels = results["labels"]

        pred_points = results["pred_points"]
        target_points = results["label_points"]

        pred_segments = results["pred_segments"]
        target_segments = results["label_segments"]

        point_dists = (pred_points.reshape(-1, 1) - target_points).abs()
        segment_dists = all_pairs_iou(pred_segments, target_segments)

        metrics = {
            "point_mae": point_dists.min(dim=0).values.float().mean().item(),
            "point_rmse": point_dists.pow(2).min(dim=0).values.float().mean().sqrt().item(),

            "segment_miou": segment_dists.max(dim=0).values.float().mean().item(),
            "segment_miou_neg": -segment_dists.max(dim=0).values.float().mean().item(),

            "pred_label_ratio": pred_labels.sum().item() / target_labels.sum().item(),
        }

        for thresh in [50, 100, 200]:
            metrics[f"point_acc@{thresh}"] = (point_dists < thresh).any(dim=0).float().mean().item()
            # m = compute_thresh_metrics(point_dists, thresh)
            # m = {f"point_{k}@{thresh}": v for k, v in m.items()}
            # metrics.update(m)

        for thresh in [0.5, 0.75, 0.9]:
            metrics[f"segment_acc@{int(thresh*100)}iou"] = (segment_dists > thresh).any(dim=0).float().mean().item()
            # m = compute_thresh_metrics(segment_dists, thresh, flip=True)
            # m = {f"segment_{k}@{int(thresh*100)}iou": v for k, v in m.items()}
            # metrics.update(m)

        return metrics


def smooth_scores(pred_scores, smoothing_window=25, smoothing_method="mean"):
    if smoothing_method not in ["mean", "max"]:
        return pred_scores

    lpad, rpad = math.floor((smoothing_window - 1) / 2), math.ceil((smoothing_window - 1) / 2)
    pred_scores = F.pad(pred_scores.unsqueeze(0), (lpad, rpad), "replicate").squeeze(0)
    pred_scores = pred_scores.unfold(0, smoothing_window, 1)
    if smoothing_method == "mean":
        pred_scores = pred_scores.mean(dim=-1)
    elif smoothing_method == "max":
        pred_scores = pred_scores.max(dim=-1).values

    return pred_scores

def find_peaks_threshold(pred_scores, quantile=0.5):
    thresh = pred_scores.quantile(quantile)
    pred_points = (pred_scores > thresh).int()
    pred_points = pred_points \
        .diff().nonzero() \
        .reshape(-1, 2).float().mean(dim=1).int()
    return pred_points

def all_pairs_iou(segments1, segments2):
    n1, n2 = segments1.size(0), segments2.size(0)

    start1 = segments1[:, 0:1].expand(-1, n2)
    end1   = segments1[:, 1:2].expand(-1, n2)
    start2 = segments2[:, 0].expand(n1, -1)
    end2   = segments2[:, 1].expand(n1, -1)

    intersection = torch.max(torch.min(end1, end2) - torch.max(start1, start2), torch.tensor(0))
    union = (end1 - start1) + (end2 - start2) - intersection
    iou = intersection / union

    return iou

def compute_thresh_metrics(dists, thresh, flip=False):
    if not flip:
        tp = (dists < thresh).any(dim=0).sum().item()
        fp = (dists > thresh).all(dim=1).sum().item()
        fn = (dists > thresh).all(dim=0).sum().item()
    else:
        tp = (dists > thresh).any(dim=0).sum().item()
        fp = (dists < thresh).all(dim=1).sum().item()
        fn = (dists < thresh).all(dim=0).sum().item()

    prec = tp / (tp + fp)
    rec = tp / (tp + fn)
    acc = tp / dists.size(1)
    f1 = 2 * (prec * rec) / (prec + rec)

    return {"acc": acc, "prec": prec, "rec": rec, "f1": f1}
