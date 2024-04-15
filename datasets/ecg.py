from abc import ABC
from pathlib import Path
import pandas as pd

from .base import (
    BaseDataset,
    ForecastDataset,
    ReconstructionDataset,
    AnomalyDetectionDataset,
    SegmentationDataset,
)


class ECGMITDataset(BaseDataset, ABC):
    supported_tasks = ["forecasting", "reconstruction", "anomaly_detection", "segmentation"]
    description = "The MIT-BIH Arrhythmia Database contains excerpts of two-channel ambulatory ECG from a mixed population of inpatients and outpatients, digitized at 360 samples per second per channel with 11-bit resolution over a 10 mV range."

    def __init__(self, config, split):
        super().__init__(config, split)
        assert self.dataset_config.version == "v2"


class ECGMITForecastingDataset(ECGMITDataset, ForecastDataset):
    def get_data(self, split=None):
        split = split or self.split

        basepath = Path(__file__).parent / "../data/mit_ecg/v2/anom/"
        split_fn = "train.csv" if split == "train" else "test.csv"
        data = pd.read_csv(basepath / split_fn)
        data = data.drop(columns=["time", "patient_id"])
        data = data.values

        return {"data": data}


class ECGMITReconstructionDataset(ECGMITDataset, ReconstructionDataset):
    def get_data(self, split=None):
        split = split or self.split

        basepath = Path(__file__).parent / "../data/mit_ecg/v2/anom/"
        split_fn = "train.csv" if split == "train" else "test.csv"
        data = pd.read_csv(basepath / split_fn)
        data = data.drop(columns=["time", "patient_id"])
        data = data.values

        return {"data": data}


class ECGMITAnomalyDetectionDataset(ECGMITDataset, AnomalyDetectionDataset):
    def get_data(self, split=None):
        split = split or self.split

        basepath = Path(__file__).parent / "../data/mit_ecg/v2/anom/"
        split_fn = "train.csv" if split == "train" else "test.csv"
        data = pd.read_csv(basepath / split_fn)

        time_col = "time"
        clip_col = "patient_id"
        feature_cols = data.columns.difference([time_col, clip_col])

        features = data[feature_cols].values
        clip_ids = data[clip_col].values.astype(int)

        if split != "train":
            labels = pd.read_csv(basepath / "test_label.csv")
            assert (labels[time_col] == data[time_col]).all()
            assert (labels[clip_col] == data[clip_col]).all()
            labels = labels.label.astype(int)
        else:
            labels = None

        desc_fn = "train_data_desc.csv" if split == "train" else "test_data_desc.csv"
        descriptions = pd.read_csv(basepath / desc_fn, index_col=0)
        descriptions = descriptions["data_desc"].to_dict()
        descriptions = {k: f"Patient description: {v}" for k, v in descriptions.items()}

        return {
            "data": features,
            "labels": labels,
            "clip_ids": clip_ids,
            "clip_descriptions": descriptions,
        }


class ECGMITSegmentationDataset(ECGMITDataset, SegmentationDataset):
    def get_data(self, split=None):
        split = split or self.split

        basepath = Path(__file__).parent / "../data/mit_ecg/v2/seg/"
        split_fn = "train.csv" if split == "train" else "test.csv"
        data = pd.read_csv(basepath / split_fn)

        time_col = "time"
        clip_col = "patient_id"
        label_col = "label"
        feature_cols = data.columns.difference([time_col, clip_col, label_col])

        features = data[feature_cols].values
        labels = data[label_col].values.astype(int)
        clip_ids = data[clip_col].values.astype(int)

        desc_fn = "train_data_desc.csv" if split == "train" else "test_data_desc.csv"
        descriptions = pd.read_csv(basepath / desc_fn, index_col=0)
        descriptions = descriptions["data_desc"].to_dict()
        descriptions = {k: f"Patient description: {v}" for k, v in descriptions.items()}

        return {"data": features, "labels": labels, "clip_ids": clip_ids, "clip_descriptions": descriptions}

    def __getitem__(self, idx):
        idx_range = self.inverse_index(idx)

        x = self.data[slice(*idx_range),:]
        y = self.labels[slice(*idx_range)]

        clip_id = self.clip_ids[idx_range[0]].item()
        desc = self.clip_descriptions[clip_id]

        return {"x_enc": x, "labels": y, "descriptions": desc}


ecg_datasets = {
    "forecasting": ECGMITForecastingDataset,
    "reconstruction": ECGMITReconstructionDataset,
    "anomaly_detection": ECGMITAnomalyDetectionDataset,
    "segmentation": ECGMITSegmentationDataset,
}
