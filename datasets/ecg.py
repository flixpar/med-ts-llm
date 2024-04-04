from abc import ABC
from pathlib import Path
import pandas as pd

from .base import BaseDataset, ForecastDataset, AnomalyDetectionDataset, SegmentationDataset


class ECGMITDataset(BaseDataset, ABC):
    supported_tasks = ["forecasting", "anomaly_detection", "segmentation"]
    description = "The MIT-BIH Arrhythmia Database contains excerpts of two-channel ambulatory ECG from a mixed population of inpatients and outpatients, digitized at 360 samples per second per channel with 11-bit resolution over a 10 mV range."


class ECGMITForecastingDataset(ECGMITDataset, ForecastDataset):
    def get_data(self, split=None):
        split = split or self.split

        basepath = Path(__file__).parent / "../data/mit_ecg/anom/"
        split_fn = "train.csv" if split == "train" else "test.csv"
        data = pd.read_csv(basepath / split_fn)
        data = data.drop(columns=["time", "patient_id"])
        data = data.values

        return {"data": data}


class ECGMITAnomalyDetectionDataset(ECGMITDataset, AnomalyDetectionDataset):
    def get_data(self, split=None):
        split = split or self.split

        basepath = Path(__file__).parent / "../data/mit_ecg/anom/"
        split_fn = "train.csv" if split == "train" else "test.csv"
        data = pd.read_csv(basepath / split_fn)
        data = data.drop(columns=["time", "patient_id"])
        data = data.values

        if split != "train":
            labels = pd.read_csv(basepath / "test_label.csv")
            labels = labels.drop(columns=["time", "patient_id"])
            labels = labels.values[:,0].astype(int)
        else:
            labels = None

        return {"data": data, "labels": labels}


class ECGMITSegmentationDataset(ECGMITDataset, SegmentationDataset):
    def get_data(self, split=None):
        split = split or self.split

        basepath = Path(__file__).parent / "../data/mit_ecg/seg/"
        split_fn = "train.csv" if split == "train" else "test.csv"
        data = pd.read_csv(basepath / split_fn)

        time_col = "time"
        clip_col = "patient_id"
        label_col = "label"
        feature_cols = data.columns.difference([time_col, clip_col, label_col])

        features = data[feature_cols].values
        labels = data[label_col].values.astype(int)
        clip_ids = data[clip_col].values.astype(int)
        timestamps = data[time_col].values

        desc_fn = "train_data_desc.csv" if split == "train" else "test_data_desc.csv"
        descriptions = pd.read_csv(basepath / desc_fn, index_col=0)
        descriptions = descriptions["data_desc"].to_dict()
        descriptions = {k: f"Patient description: {v}" for k, v in descriptions.items()}

        return {"data": features, "labels": labels, "clip_ids": clip_ids, "timestamps": timestamps, "clip_descriptions": descriptions}

    def __getitem__(self, idx):
        idx = idx * self.step_size
        idx_range = (idx, idx + self.pred_len)

        x = self.data[slice(*idx_range),:]
        y = self.labels[slice(*idx_range)]

        clip_id = self.clip_ids[idx]
        desc = self.clip_descriptions[clip_id]

        return {"x_enc": x, "labels": y, "descriptions": desc}


ecg_datasets = {
    "forecasting": ECGMITForecastingDataset,
    "anomaly_detection": ECGMITAnomalyDetectionDataset,
    "segmentation": ECGMITSegmentationDataset,
}
