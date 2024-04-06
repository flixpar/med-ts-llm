import torch
import torch.nn.functional as F

import math
import scipy

from bayes_opt import BayesianOptimization

from tqdm import tqdm

from .base import BaseTask


class SegmentationTask(BaseTask):

    def __init__(self, run_id, config, newrun=True):
        self.task = "segmentation"
        self.segmentation_mode = config.tasks.segmentation.mode
        super(SegmentationTask, self).__init__(run_id, config, newrun)

    def train(self):
        for epoch in range(self.config.training.epochs):
            print(f"Epoch {epoch + 1}/{self.config.training.epochs}")
            self.model.train()
            for inputs in tqdm(self.train_dataloader):
                inputs = self.prepare_batch(inputs)

                with torch.autocast(self.device.type, dtype=torch.bfloat16, enabled=self.mixed):
                    pred = self.model(inputs)
                    loss = self.loss_fn(pred, inputs["labels"].to(self.dtype))

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
                assert self.config.tasks.segmentation.mode == "boundary-prediction"
                self.loss_fn = torch.nn.BCEWithLogitsLoss()
            case "mse":
                assert self.config.tasks.segmentation.mode == "steps-to-boundary"
                self.loss_fn = torch.nn.MSELoss()
            case "mae":
                assert self.config.tasks.segmentation.mode == "steps-to-boundary"
                self.loss_fn = torch.nn.L1Loss()
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

        target_dtype = torch.int if self.segmentation_mode == "boundary-prediction" else torch.float

        preds = torch.full((n_points,), float("nan"))
        targets = torch.full((n_points,), -1, dtype=target_dtype)

        with torch.no_grad():
            for idx, inputs in tqdm(enumerate(dataloader), total=len(dataloader)):
                inputs = self.prepare_batch(inputs)
                pred = self.model(inputs)

                for j in range(pred.size(0)):
                    inds = dataset.inverse_index((idx * bs) + j)
                    time_inds = slice(*(inds[0] if dataset.univariate else inds))

                    preds[time_inds] = pred[j].squeeze().cpu().detach()
                    targets[time_inds] = inputs["labels"][j].squeeze().cpu().detach()

        if step_size > pred_len:
            cutoff = n_points - (n_points % step_size)
            preds, targets = preds[:cutoff], targets[:cutoff]
            preds = preds.reshape(-1, step_size)[:, :pred_len].reshape(-1)
            targets = targets.reshape(-1, step_size)[:, :pred_len].reshape(-1)

        assert not torch.isnan(preds).any()
        assert not (targets < 0).any()

        if self.segmentation_mode == "boundary-prediction":
            return self.process_preds_boundary_prediction(preds, targets)
        elif self.segmentation_mode == "steps-to-boundary":
            return self.process_preds_steps_to_boundary(preds, targets)
        else:
            raise ValueError(f"Segmentation mode {self.segmentation_mode} not supported")

    def process_preds_boundary_prediction(self, preds, targets):
        pred_scores = preds.clone()

        # pred_scores = smooth_scores(pred_scores, 25, "mean")
        # pred_points = find_peaks_threshold(pred_scores, 0.55)

        if self.config.tasks.segmentation.distance_thresh == "auto":
            # distance_thresh = 0.75 * targets.size(0) / targets.sum().item()
            target_seg_lens = targets.nonzero().squeeze().unfold(0, 2, 1).diff(dim=1).squeeze()
            distance_thresh = target_seg_lens.float().quantile(0.1).item()
        elif self.config.tasks.segmentation.distance_thresh == "optimize":
            est = targets.size(0) / targets.sum().item()
            distance_thresh = optimize_threshold(pred_scores, targets, est)
        else:
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

    def process_preds_steps_to_boundary(self, preds, targets):
        pred_scores = preds.clone()

        targets = (targets == 0).int()
        threshold_est = targets.size(0) / targets.sum().item()

        pts_max = scipy.signal.find_peaks(pred_scores.numpy(), prominence=0.5)[0]
        pts_min = scipy.signal.find_peaks(-pred_scores.numpy(), prominence=0.5)[0]
        pts_a, pts_b = (pts_max, pts_min) if len(pts_max) >= len(pts_min) else (pts_min, pts_max)
        pts_a, pts_b = torch.tensor(pts_a), torch.tensor(pts_b)

        pred_points = torch.empty_like(pts_a)
        for idx, pt in enumerate(pts_a):
            dists = (pts_b - pt).abs()
            closest_idx = dists.argmin().item()
            pred_points[idx] = pt if dists[closest_idx] > threshold_est/2 else pts_b[closest_idx]

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

        if len(pred_points) == 0:
            return {
                "point_mae": float("inf"),
                "point_rmse": float("inf"),
                "segment_miou": 0,
                "pred_label_ratio": 0.0,
            }

        metrics = {
            "point_mae": point_dists.min(dim=0).values.float().mean().item(),
            "point_rmse": point_dists.pow(2).min(dim=0).values.float().mean().sqrt().item(),

            "segment_miou": segment_dists.max(dim=0).values.float().mean().item(),

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

def optimize_threshold(pred_scores, targets, est):
    pred_scores = pred_scores.numpy()

    target_points = targets.nonzero().squeeze()
    target_segments = torch.cat([torch.tensor([0]), target_points, torch.tensor([len(pred_scores)-1])])
    target_segments = target_segments.unfold(0, 2, 1)

    def score_fn(thresh):
        pred_points = scipy.signal.find_peaks(pred_scores, distance=thresh)[0]

        pred_segments = torch.cat([
            torch.tensor([0], dtype=torch.int),
            torch.tensor(pred_points, dtype=torch.int),
            torch.tensor([len(pred_scores)-1], dtype=torch.int),
        ])
        pred_segments = pred_segments.unfold(0, 2, 1)

        segment_dists = all_pairs_iou(pred_segments, target_segments)
        seg_miou = segment_dists.max(dim=0).values.float().mean().item()
        return seg_miou

    optimizer = BayesianOptimization(
        f = score_fn,
        pbounds = {"thresh": (0.5 * est, 1.25 * est)},
        random_state = 0,
        verbose = 0,
        allow_duplicate_points=True,
    )
    optimizer.maximize(init_points=5, n_iter=10)
    return optimizer.max["params"]["thresh"]
