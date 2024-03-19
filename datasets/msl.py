from abc import ABC
from pathlib import Path

import numpy as np

from torch.utils.data import Dataset


class MSLDataset(Dataset, ABC):
    def __init__(self, config, split):
        assert config.data.cols == "all"
        assert config.data.normalize == False

        self.split = split
        self.task = config.task
        self.history_len = config.history_len
        self.pred_len = config.pred_len
        self.step_size = config.data.step

        if self.split == "test":
            self.step_size = self.pred_len

        basepath = Path(__file__).parent / "../data/msl/"
        split_fn = "MSL_train.npy" if split == "train" else "MSL_test.npy"
        self.data = np.load(basepath / split_fn)

        if config.data.normalize:
            raise NotImplementedError("Normalization not implemented")

        self.n_points = self.data.shape[0]
        self.n_features = self.data.shape[1]
        self.mode = "multivariate"

        self.description = "The MSL (Mars Science Laboratory rover) dataset was created by NASA and consists of telemetry data across 55 sensors on the rover. The data is collected at 1 minute intervals and spans a period of 78 Martian days. The dataset is labeled with 143 anomalous intervals, each of which is labeled by an expert as an incident, surprise, or an anomaly."

        self.supported_tasks = ["forecasting", "anomaly_detection"]
        assert self.task in self.supported_tasks


class MSLForecastingDataset(MSLDataset):
    def __init__(self, config, split):
        super(MSLForecastingDataset, self).__init__(config, split)
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


class MSLAnomalyDetectionDataset(MSLDataset):
    def __init__(self, config, split):
        super(MSLAnomalyDetectionDataset, self).__init__(config, split)
        assert self.task == "anomaly_detection"

        if self.split != "train":
            basepath = Path(__file__).parent / "../data/msl/"
            labels = np.load(basepath / "MSL_test_label.npy")
            self.labels = labels.astype(int)
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


def MSLDatasetSelector(config, split):
    if config.task == "forecasting":
        return MSLForecastingDataset(config, split)
    elif config.task == "anomaly_detection":
        return MSLAnomalyDetectionDataset(config, split)
    else:
        raise ValueError(f"Task {config.task} not supported by dataset {config.data.dataset}")
