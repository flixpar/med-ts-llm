from abc import ABC
from pathlib import Path

import pandas as pd
from sklearn.preprocessing import StandardScaler

from torch.utils.data import Dataset


class BIDMCDataset(Dataset, ABC):

    supported_tasks = ["forecasting", "segmentation"]

    def __init__(self, config, split):
        assert config.data.cols == "all"
        assert config.task in self.supported_tasks

        self.split = split
        self.task = config.task
        self.history_len = config.history_len
        self.pred_len = config.pred_len
        self.step_size = config.data.step

        if self.split == "test":
            self.step_size = self.pred_len

        basepath = Path(__file__).parent / "../data/bidmc/"
        split_fn = "train.csv" if split == "train" else "test.csv"
        data = pd.read_csv(basepath / split_fn)

        time_col = "Time"
        clip_col = "patient_id"
        label_col = "label"
        feature_cols = data.columns.difference([time_col, clip_col, label_col])

        self.data = data[feature_cols].values
        self.labels = data[label_col].values.astype(int)
        self.clip_ids = data[clip_col].values.astype(int)
        self.timestamps = data[time_col].values

        if config.data.normalize:
            train_data = pd.read_csv(basepath / "train.csv")[feature_cols].values
            self.normalizer = StandardScaler().fit(train_data)
            self.data = self.normalizer.transform(self.data)

        self.n_points, self.n_features = self.data.shape
        self.mode = "multivariate"

        self.description = "The BIDMC dataset is a dataset of electrocardiogram (ECG), pulse oximetry (photoplethysmogram, PPG) and impedance pneumography respiratory signals acquired from intensive care patients. Two annotators manually annotated individual breaths in each recording using the impedance respiratory signal."


class BIDMCForecastingDataset(BIDMCDataset):
    def __init__(self, config, split):
        super(BIDMCForecastingDataset, self).__init__(config, split)
        assert self.task == "forecasting"

    def __getitem__(self, idx):
        idx = idx * self.step_size
        x_range = (idx, idx + self.history_len)
        y_range = (x_range[1], x_range[1] + self.pred_len)

        x = self.data[slice(*x_range),:]
        y = self.data[slice(*y_range),:]

        return {"x_enc": x, "y": y}

    def __len__(self):
        return (self.n_points - self.history_len - self.pred_len + 1) // self.step_size
    
    def inverse_index(self, idx):
        return idx * self.step_size + self.history_len


class BIDMCSegmentationDataset(BIDMCDataset):
    def __init__(self, config, split):
        super(BIDMCSegmentationDataset, self).__init__(config, split)
        assert self.task == "segmentation"
        assert self.pred_len == self.history_len

    def __getitem__(self, idx):
        idx = idx * self.step_size
        idx_range = (idx, idx + self.pred_len)

        x = self.data[slice(*idx_range),:]
        y = self.labels[slice(*idx_range)]

        return {"x_enc": x, "labels": y}

    def __len__(self):
        return (self.n_points - self.pred_len) // self.step_size + 1

    def inverse_index(self, idx):
        return idx * self.step_size


def BIDMCDatasetSelector(config, split):
    if config.task == "forecasting":
        return BIDMCForecastingDataset(config, split)
    elif config.task == "segmentation":
        return BIDMCSegmentationDataset(config, split)
    else:
        raise ValueError(f"Task {config.task} not supported by dataset {config.data.dataset}")
