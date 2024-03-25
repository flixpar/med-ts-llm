from abc import ABC
from pathlib import Path

import pandas as pd
from sklearn.preprocessing import StandardScaler

from torch.utils.data import Dataset


class VentilatorDataset(Dataset, ABC):

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

        self.mode = "multivariate"

        self.description = "The dataset contains time-series data of airway pressure and flow rate measurements collected from a mechanical ventilator during the respiratory support of a fully sedated patient. The data is sampled at a frequency of 100 Hz. The airway pressure is measured in cmH2O and the flow rate is measured in L/min."


class VentilatorForecastingDataset(VentilatorDataset):
    def __init__(self, config, split):
        super(VentilatorForecastingDataset, self).__init__(config, split)
        assert self.task == "forecasting"

        basepath = Path(__file__).parent / "../data/ventilator/v1/"
        waveform_files = basepath.glob("*.csv")
        waveform_files = sorted(waveform_files)
        dfs = [pd.read_csv(f, usecols=["pressure", "flow"]) for f in waveform_files]
        data = pd.concat(dfs, ignore_index=True).values

        train_pct, val_pct, test_pct = 0.7, 0.15, 0.15
        train_idx = int(train_pct * data.shape[0])
        val_idx = int((train_pct + val_pct) * data.shape[0])
        train = data[:train_idx,:]

        match split:
            case "train":
                self.data = train
            case "val":
                self.data = data[train_idx:val_idx,:]
            case "test":
                self.data = data[val_idx:,:]
            case _:
                raise ValueError(f"Invalid split: {split}")

        if config.data.normalize:
            self.normalizer = StandardScaler().fit(train)
            self.data = self.normalizer.transform(self.data)

        self.n_points = self.data.shape[0]
        self.n_features = self.data.shape[1]

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


class VentilatorSemanticSegmentationDataset(VentilatorDataset):
    def __init__(self, config, split):
        super(VentilatorSemanticSegmentationDataset, self).__init__(config, split)
        assert self.task == "semantic_segmentation"
        assert self.pred_len == self.history_len

        basepath = Path(__file__).parent / "../data/ventilator/v3/"
        waveform_files = basepath.glob("*.csv")
        waveform_files = [fn for fn in waveform_files if fn.name != "patient_704_vent_w_1_labeled.csv"]
        waveform_files = sorted(waveform_files)
        dfs = [pd.read_csv(f, usecols=["pressure", "flow", "label"]) for f in waveform_files]
        dfs = [df[df.label >= 0] for df in dfs]
        dfs = [df for df in dfs if len(df) > 1000]
        data = pd.concat(dfs, ignore_index=True)

        labels = data["label"].values.astype(int)
        data = data[["pressure", "flow"]].values

        train_pct, val_pct, test_pct = 0.7, 0.15, 0.15
        train_idx = int(train_pct * data.shape[0])
        val_idx = int((train_pct + val_pct) * data.shape[0])
        train = data[:train_idx]

        match split:
            case "train":
                self.data = train
                self.labels = labels[:train_idx]
            case "val":
                self.data = data[train_idx:val_idx]
                self.labels = labels[train_idx:val_idx]
            case "test":
                self.data = data[val_idx:]
                self.labels = labels[val_idx:]
            case _:
                raise ValueError(f"Invalid split: {split}")

        if config.data.normalize:
            self.normalizer = StandardScaler().fit(train)
            self.data = self.normalizer.transform(self.data)

        self.n_points = self.data.shape[0]
        self.n_features = self.data.shape[1]
        self.n_classes = 2

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


def VentilatorDatasetSelector(config, split):
    if config.task == "forecasting":
        return VentilatorForecastingDataset(config, split)
    elif config.task == "semantic_segmentation":
        return VentilatorSemanticSegmentationDataset(config, split)
    else:
        raise ValueError(f"Task {config.task} not supported by dataset {config.data.dataset}")
