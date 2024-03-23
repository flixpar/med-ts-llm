from abc import ABC
from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

from torch.utils.data import Dataset


class LUDBDataset(Dataset, ABC):

    supported_tasks = ["forecasting", "semantic_segmentation"]

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

        basepath = Path(__file__).parent / "../data/ludb/"
        split_fn = "train.csv" if split == "train" else "test.csv"
        data = pd.read_csv(basepath / split_fn)

        self.data = data["ecg"].values[:,np.newaxis]
        self.labels = data["label"].values.astype(int)

        if config.data.normalize:
            train_data = self.data if split == "train" else pd.read_csv(basepath / "train.csv", usecols=["ecg"]).values
            self.normalizer = StandardScaler().fit(train_data)
            self.data = self.normalizer.transform(self.data)

        self.n_points, self.n_features = self.data.shape
        self.n_classes = len(np.unique(self.labels))
        self.mode = "multivariate"

        self.description = "LUDB is an ECG signal database with marked boundaries and peaks of P, T waves and QRS complexes. Cardiologists manually annotated all 200 records of healthy and sick patients which contains a corresponding diagnosis. This can be used for ECG delineation."


class LUDBForecastingDataset(LUDBDataset):
    def __init__(self, config, split):
        super(LUDBForecastingDataset, self).__init__(config, split)
        assert self.task == "forecasting"

    def __getitem__(self, idx):
        idx = idx * self.step_size
        x_range = (idx, idx + self.history_len)
        y_range = (x_range[1], x_range[1] + self.pred_len)

        x = self.data[slice(*x_range),:]
        y = self.data[slice(*y_range),:]

        x_dec = x[0:0,:]

        return x, x_dec, y

    def __len__(self):
        return (self.n_points - self.history_len - self.pred_len + 1) // self.step_size

    def inverse_index(self, idx):
        return idx * self.step_size + self.history_len


class LUDBSemanticSegmentationDataset(LUDBDataset):
    def __init__(self, config, split):
        super(LUDBSemanticSegmentationDataset, self).__init__(config, split)
        assert self.task == "semantic_segmentation"
        assert self.pred_len == self.history_len

    def __getitem__(self, idx):
        idx = idx * self.step_size
        idx_range = (idx, idx + self.pred_len)

        x = self.data[slice(*idx_range),:]
        y = self.labels[slice(*idx_range)]
        x_dec = x[0:0,:]

        return x, x_dec, y

    def __len__(self):
        return (self.n_points - self.pred_len) // self.step_size + 1

    def inverse_index(self, idx):
        return idx * self.step_size


def LUDBDatasetSelector(config, split):
    if config.task == "forecasting":
        return LUDBForecastingDataset(config, split)
    elif config.task == "semantic_segmentation":
        return LUDBSemanticSegmentationDataset(config, split)
    else:
        raise ValueError(f"Task {config.task} not supported by dataset {config.data.dataset}")
