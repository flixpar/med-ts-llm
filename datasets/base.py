from abc import ABC, abstractmethod

import torch
import numpy as np

from sklearn.preprocessing import StandardScaler

from torch.utils.data import Dataset


class BaseDataset(Dataset, ABC):

    data = None
    labels = None
    timestamps = None
    clip_ids = None
    clip_descriptions = None

    normalizer = None
    univariate = False
    clip_dataset = False

    supported_tasks = []

    def __init__(self, config, split):
        super().__init__()

        self.config = config
        self.split = split
        self.task = config.task
        self.name = config.data.dataset

        self.task_config = self.config.get("tasks", {}).get(self.task, {})
        self.dataset_config = self.config.get("datasets", {}).get(self.name, {})
        self.data_config = self.config.data

        self.history_len = config.history_len
        self.pred_len = config.pred_len
        self.step_size = config.data.step

        if self.split == "test":
            self.step_size = self.pred_len

        assert config.data.cols == "all"
        assert config.task in self.supported_tasks

        self.load_data()

    @abstractmethod
    def __len__(self):
        raise NotImplementedError()

    @abstractmethod
    def __getitem__(self, idx):
        raise NotImplementedError()

    @abstractmethod
    def inverse_index(self, idx):
        raise NotImplementedError()

    def load_data(self):
        data = self.get_data()
        if "data" in data:
            self.data = data["data"]
            self.data = self.normalize(self.data)
            self.data = torch.tensor(self.data, dtype=torch.float32)
        if "labels" in data and data["labels"] is not None:
            n_labels = len(np.unique(data["labels"]))
            int_dtype = torch.long if (n_labels > 2) else torch.int32
            self.labels = torch.tensor(data["labels"], dtype=int_dtype)
        if "timestamps" in data:
            self.timestamps = torch.tensor(data["timestamps"], dtype=torch.float32)
        if "clip_ids" in data:
            self.clip_ids = torch.tensor(data["clip_ids"], dtype=torch.int32)
        if "clip_descriptions" in data:
            self.clip_descriptions = data["clip_descriptions"]

    @abstractmethod
    def get_data(self, split=None):
        pass

    def normalize(self, data):
        if not self.config.data.normalize:
            return data
        if self.normalizer is not None:
            return self.normalizer.transform(data)

        train_data = self.data if (self.split == "train") else self.get_data("train")["data"]
        self.normalizer = StandardScaler().fit(train_data)
        return self.normalizer.transform(data)

    def denormalize(self, data):
        return self.normalizer.inverse_transform(data)

    @property
    def n_points(self):
        return self.data.shape[0]

    @property
    def n_features(self):
        return self.data.shape[1]

    @property
    def n_classes(self):
        return 0

    @property
    def real_features(self):
        return self.n_features

    @property
    def description(self):
        return self.__doc__


class ForecastDataset(BaseDataset, ABC):

    def __init__(self, config, split):
        super().__init__(config, split)
        assert self.task == "forecasting"

    def __getitem__(self, idx):
        x_range, y_range = self.inverse_index(idx)

        x = self.data[slice(*x_range),:]
        y = self.data[slice(*y_range),:]

        return {"x_enc": x, "y": y}

    def __len__(self):
        return (self.n_points - self.history_len - self.pred_len + 1) // self.step_size

    def inverse_index(self, idx):
        idx = idx * self.step_size
        x_range = (idx, idx + self.history_len)
        y_range = (x_range[1], x_range[1] + self.pred_len)
        return x_range, y_range


class ReconstructionDataset(BaseDataset, ABC):

    def __init__(self, config, split):
        super().__init__(config, split)
        assert self.task == "reconstruction"
        assert self.pred_len == self.history_len

    def __getitem__(self, idx):
        x_range = self.inverse_index(idx)
        x = self.data[slice(*x_range),:]
        return {"x_enc": x}

    def __len__(self):
        return (self.n_points - self.pred_len) // self.step_size + 1

    def inverse_index(self, idx):
        idx = idx * self.step_size
        x_range = (idx, idx + self.pred_len)
        return x_range


class AnomalyDetectionDataset(BaseDataset, ABC):

    def __init__(self, config, split):
        super().__init__(config, split)
        assert self.task == "anomaly_detection"
        assert self.pred_len == self.history_len

    def __getitem__(self, idx):
        x_range = self.inverse_index(idx)
        x = self.data[slice(*x_range),:]

        if self.labels is not None:
            labels = self.labels[slice(*x_range)]
        else:
            labels = x[0:0,0]

        return {"x_enc": x, "labels": labels}

    def __len__(self):
        return (self.n_points - self.pred_len) // self.step_size + 1

    def inverse_index(self, idx):
        idx = idx * self.step_size
        x_range = (idx, idx + self.pred_len)
        return x_range


class SemanticSegmentationDataset(BaseDataset, ABC):

    def __init__(self, config, split):
        super().__init__(config, split)
        assert self.task == "semantic_segmentation"
        assert self.pred_len == self.history_len

    def __getitem__(self, idx):
        idx_range = self.inverse_index(idx)

        x = self.data[slice(*idx_range),:]
        y = self.labels[slice(*idx_range)]

        return {"x_enc": x, "labels": y}

    def __len__(self):
        return (self.n_points - self.pred_len) // self.step_size + 1

    def inverse_index(self, idx):
        idx = idx * self.step_size
        idx_range = (idx, idx + self.pred_len)
        return idx_range

    @property
    def n_classes(self):
        return len(self.labels.unique())


class SegmentationDataset(BaseDataset, ABC):

    def __init__(self, config, split):
        super().__init__(config, split)
        assert self.task == "segmentation"
        assert self.pred_len == self.history_len
        self.convert_labels()

    def __getitem__(self, idx):
        idx_range = self.inverse_index(idx)

        x = self.data[slice(*idx_range),:]
        y = self.labels[slice(*idx_range)]

        return {"x_enc": x, "labels": y}

    def __len__(self):
        return (self.n_points - self.pred_len) // self.step_size + 1

    def inverse_index(self, idx):
        idx = idx * self.step_size
        idx_range = (idx, idx + self.pred_len)
        return idx_range

    def convert_labels(self):
        if self.task_config.mode == "steps-to-boundary":
            labels_binary = self.labels.numpy()
            changepts = np.where(labels_binary)[0]
            changepts = np.append(changepts, len(labels_binary))
            labels = np.zeros(len(labels_binary), dtype=np.float32)
            seg_len = changepts[0]
            for i in range(len(labels)):
                labels[i] = (changepts[0] - i) / seg_len
                if i == changepts[0]:
                    changepts = changepts[1:]
                    seg_len = changepts[0] - i
            self.labels = torch.tensor(labels)
        elif self.task_config.mode == "boundary-prediction":
            pass
        else:
            raise ValueError(f"Segmentation mode {self.task_config.mode} not supported")


class ClipDataset(BaseDataset, ABC):
    clip_dataset = True

    def __init__(self, config, split):
        super().__init__(config, split)

        assert not self.task == "forecasting", "ClipDataset does not support forecasting"

        assert self.clip_ids is not None
        assert (self.clip_ids.diff() >= 0).all()

        clips, self.clip_inds, self.clip_lens = self.clip_ids.unique_consecutive(return_inverse=True, return_counts=True)
        self.clips = torch.arange(len(clips))

        assert (clips == self.clip_ids.unique()).all()

        self.clip_lens_cumsum = self.clip_lens.cumsum(0)
        self.clip_lens_cumsum = torch.cat([torch.tensor([0]), self.clip_lens_cumsum])

        self.clip_segs = (self.clip_lens - self.pred_len) // self.step_size + 1

        self.clip_segs_cumsum = self.clip_segs.cumsum(0)
        self.clip_segs_cumsum = torch.cat([torch.tensor([0]), self.clip_segs_cumsum])

        self.dataset_len = self.clip_segs_cumsum[-1].item()

        clip_pts = ((self.clip_segs - 1) * self.step_size) + self.pred_len
        clip_remainder = self.clip_lens - clip_pts
        assert (clip_remainder >= 0).all()

        clip_mask = ((torch.arange(clip_pts.max()) % self.step_size) // self.pred_len) == 0
        mask_1s = [clip_mask[:l] for l in clip_pts]
        mask_0s = [torch.zeros(r, dtype=torch.bool) for r in clip_remainder]
        mask = [torch.cat([m1, m0]) for m1, m0 in zip(mask_1s, mask_0s)]
        self.mask = torch.cat(mask)

        assert len(self.mask) == self.n_points

    def __len__(self):
        return self.dataset_len

    def inverse_index(self, seg_idx):
        clip_idx = torch.searchsorted(self.clip_segs_cumsum, seg_idx, right=True).item() - 1

        clip_seg_idx = seg_idx - self.clip_segs_cumsum[clip_idx].item()
        clip_start = self.clip_lens_cumsum[clip_idx].item()

        seg_start = clip_start + (clip_seg_idx * self.step_size)
        seg_end = seg_start + self.pred_len
        seg_range = (seg_start, seg_end)

        return seg_range
