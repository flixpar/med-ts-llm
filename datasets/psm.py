from pathlib import Path

import pandas as pd
import numpy as np

from torch.utils.data import Dataset


class PSMDataset(Dataset):
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

        basepath = Path(__file__).parent / "../data/psm/"
        split_fn = "train.csv" if split == "train" else "test.csv"
        data = pd.read_csv(basepath / split_fn)
        data = data.drop(columns=["timestamp_(min)"])

        self.data = np.nan_to_num(data.values)

        if self.split != "train" and self.task == "anomaly_detection":
            labels = pd.read_csv(basepath / "test_label.csv")
            labels = labels.drop(columns=["timestamp_(min)"])
            self.labels = labels.values[:,0].astype(int)
        else:
            self.labels = None

        if config.data.normalize:
            raise NotImplementedError("Normalization not implemented")

        self.n_features = self.data.shape[1]
        self.mode = "multivariate"

        self.description = "The PSM dataset is proposed by eBay and consists of 26 dimensional data captured internally from application server nodes. The dataset is used to predict the number of sessions in the next 10 minutes based on the current and historical data."

        self.supported_tasks = ["forecasting", "anomaly_detection"]
        assert self.task in self.supported_tasks

    def __len__(self):
        if self.task == "forecasting":
            return ((self.data.shape[0] - self.history_len - self.pred_len) // self.step_size) + 1
        elif self.task == "anomaly_detection":
            return ((self.data.shape[0] - self.pred_len) // self.step_size) + 1
        else:
            return 0

    def __getitem__(self, idx):
        if self.task == "forecasting":
            return self.getitem_forecast(idx)
        elif self.task == "anomaly_detection":
            return self.getitem_anomaly_detection(idx)
        else:
            return None

    def getitem_forecast(self, idx):
        idx = idx * self.step_size
        x_range = (idx, idx + self.history_len)
        y_range = (x_range[1], x_range[1] + self.pred_len)

        x = self.data[slice(*x_range),:]
        y = self.data[slice(*y_range),:]

        x_dec = x[0:0,:]

        return x, x_dec, y

    def getitem_anomaly_detection(self, idx):
        idx = idx * self.step_size
        x_range = (idx, idx + self.pred_len)
        x = self.data[slice(*x_range),:]

        x_dec = x[0:0,:]

        if self.labels is not None:
            labels = self.labels[slice(*x_range)]
        else:
            labels = x[0:0,0]

        return x, x_dec, labels
