from abc import ABC
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from torch.utils.data import Dataset


class ECGMITDataset(Dataset, ABC):
    def __init__(self, config, split):
        assert config.data.cols == "all"

        self.split = split
        self.task = config.task
        self.history_len = config.history_len
        self.pred_len = config.pred_len
        self.step_size = config.data.step

        if self.split == "test":
            self.step_size = self.pred_len

        basepath = Path(__file__).parent / "../data/mit_ecg/anom/"
        split_fn = "train.csv" if split == "train" else "test.csv"
        data = pd.read_csv(basepath / split_fn)
        data = data.drop(columns=["time", "patient_id"])
        self.data = data.values

        if config.data.normalize:
            self.normalizer = StandardScaler()
            self.data = self.normalizer.fit_transform(self.data)

        self.n_points = self.data.shape[0]
        self.n_features = self.data.shape[1]
        self.mode = "multivariate"

        self.description = "The MIT-BIH Arrhythmia Database contains excerpts of two-channel ambulatory ECG from a mixed population of inpatients and outpatients, digitized at 360 samples per second per channel with 11-bit resolution over a 10 mV range."

        self.supported_tasks = ["forecasting", "anomaly_detection", "segmentation"]
        assert self.task in self.supported_tasks


class ECGMITForecastingDataset(ECGMITDataset):
    def __init__(self, config, split):
        super(ECGMITForecastingDataset, self).__init__(config, split)
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


class ECGMITAnomalyDetectionDataset(ECGMITDataset):
    def __init__(self, config, split):
        super(ECGMITAnomalyDetectionDataset, self).__init__(config, split)
        assert self.task == "anomaly_detection"

        if self.split != "train":
            basepath = Path(__file__).parent / "../data/mit_ecg/anom/"
            labels = pd.read_csv(basepath / "test_label.csv")
            labels = labels.drop(columns=["time", "patient_id"])
            self.labels = labels.values[:,0].astype(int)
        else:
            self.labels = None

    def __getitem__(self, idx):
        idx = idx * self.step_size
        x_range = (idx, idx + self.pred_len)
        x = self.data[slice(*x_range),:]

        if self.labels is not None:
            labels = self.labels[slice(*x_range)]
        else:
            labels = x[0:0,0]

        return {"x_enc": x, "labels": labels}

    def __len__(self):
        return (self.n_points - self.pred_len) // self.step_size + 1

    def inverse_index(self, idx):
        return idx * self.step_size


class ECGMITSegmentationDataset(ECGMITDataset):
    def __init__(self, config, split):
        super(ECGMITSegmentationDataset, self).__init__(config, split)
        assert self.task == "segmentation"

        basepath = Path(__file__).parent / "../data/mit_ecg/seg/"
        split_fn = "train.csv" if split == "train" else "test.csv"
        data = pd.read_csv(basepath / split_fn)

        time_col = "time"
        clip_col = "patient_id"
        label_col = "label"
        feature_cols = data.columns.difference([time_col, clip_col, label_col])

        self.data = data[feature_cols].values
        self.clip_ids = data[clip_col].values.astype(int)
        self.timestamps = data[time_col].values

        if config.data.normalize:
            train_data = pd.read_csv(basepath / "train.csv")[feature_cols].values
            self.normalizer = StandardScaler().fit(train_data)
            self.data = self.normalizer.transform(self.data)

        if config.tasks.segmentation.mode == "steps-to-boundary":
            labels_binary = data[label_col].values.astype(int)
            changepts = np.where(labels_binary)[0]
            changepts = np.append(changepts, len(labels_binary))
            labels = np.zeros(len(labels_binary), dtype=np.float32)
            seg_len = changepts[0]
            for i in range(len(labels)):
                labels[i] = (changepts[0] - i) / seg_len
                if i == changepts[0]:
                    changepts = changepts[1:]
                    seg_len = changepts[0] - i
            self.labels = labels
        elif config.tasks.segmentation.mode == "boundary-prediction":
            self.labels = data[label_col].values.astype(int)
        else:
            raise ValueError(f"Segmentation mode {config.tasks.segmentation.mode} not supported")

        self.n_points, self.n_features = self.data.shape

        desc_fn = "train_data_desc.csv" if split == "train" else "test_data_desc.csv"
        descriptions = pd.read_csv(basepath / desc_fn, index_col=0)
        self.descriptions = descriptions["data_desc"].to_dict()

    def __getitem__(self, idx):
        idx = idx * self.step_size
        idx_range = (idx, idx + self.pred_len)

        x = self.data[slice(*idx_range),:]
        y = self.labels[slice(*idx_range)]

        patient_id = self.clip_ids[idx]
        desc = self.descriptions[patient_id]

        return {"x_enc": x, "labels": y, "descriptions": f"Patient description: {desc}"}

    def __len__(self):
        return (self.n_points - self.pred_len) // self.step_size + 1

    def inverse_index(self, idx):
        return idx * self.step_size


def ECGMITDatasetSelector(config, split):
    if config.task == "forecasting":
        return ECGMITForecastingDataset(config, split)
    elif config.task == "anomaly_detection":
        return ECGMITAnomalyDetectionDataset(config, split)
    elif config.task == "segmentation":
        return ECGMITSegmentationDataset(config, split)
    else:
        raise ValueError(f"Task {config.task} not supported by dataset {config.data.dataset}")
