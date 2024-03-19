from abc import ABC
from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

from torch.utils.data import Dataset


class PSMDataset(Dataset, ABC):
    def __init__(self, config, split):
        assert config.data.cols == "all"

        self.split = split
        self.task = config.task
        self.history_len = config.history_len
        self.pred_len = config.pred_len
        self.step_size = config.data.step

        if self.split == "test":
            self.step_size = self.pred_len

        basepath = Path(__file__).parent / "../data/psm/"
        split_fn = "train.csv" if split == "train" else "test.csv"
        data = pd.read_csv(basepath / split_fn)
        data = data.drop(columns=["timestamp_(min)"])

        self.data = np.nan_to_num(data.values)

        if config.data.normalize:
            self.normalizer = StandardScaler()
            self.data = self.normalizer.fit_transform(self.data)

        self.n_points = self.data.shape[0]
        self.n_features = self.data.shape[1]
        self.mode = "multivariate"

        self.description = "The PSM dataset is proposed by eBay and consists of 26 dimensional data captured internally from application server nodes. The dataset is used to predict the number of sessions in the next 10 minutes based on the current and historical data."

        self.supported_tasks = ["forecasting", "anomaly_detection"]
        assert self.task in self.supported_tasks


class PSMForecastingDataset(PSMDataset):
    def __init__(self, config, split):
        super(PSMForecastingDataset, self).__init__(config, split)
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


class PSMAnomalyDetectionDataset(PSMDataset):
    def __init__(self, config, split):
        super(PSMAnomalyDetectionDataset, self).__init__(config, split)
        assert self.task == "anomaly_detection"

        if self.split != "train":
            basepath = Path(__file__).parent / "../data/psm/"
            labels = pd.read_csv(basepath / "test_label.csv")
            labels = labels.drop(columns=["timestamp_(min)"])
            self.labels = labels.values[:,0].astype(int)
        else:
            self.labels = None

    def __getitem__(self, idx):
        idx = idx * self.step_size
        x_range = (idx, idx + self.pred_len)
        x = self.data[slice(*x_range),:]

        x_dec = x[0:0,:]

        if self.labels is not None:
            labels = self.labels[slice(*x_range)]
        else:
            labels = x[0:0,0]

        return x, x_dec, labels

    def __len__(self):
        return self.n_points // self.step_size

    def inverse_index(self, idx):
        return idx * self.step_size


def PSMDatasetSelector(config, split):
    if config.task == "forecasting":
        return PSMForecastingDataset(config, split)
    elif config.task == "anomaly_detection":
        return PSMAnomalyDetectionDataset(config, split)
    else:
        raise ValueError(f"Task {config.task} not supported by dataset {config.data.dataset}")
