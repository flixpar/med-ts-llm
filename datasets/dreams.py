from abc import ABC
from pathlib import Path
import pandas as pd

from .base import (
    BaseDataset,
    ForecastDataset,
    ReconstructionDataset,
    AnomalyDetectionDataset,
)


class DreamsDataset(BaseDataset, ABC):
    supported_tasks = ["forecasting", "reconstruction", "anomaly_detection"]
    description = "The DREAMS database consists of digital 32-channel polysomnographic recordings (PSG), acquired from patients with different pathologies in a sleep hospital laboratory. Muscle or movement artifacts on the electroencephalogram (EEG) were annotated in microevents or in sleep stages by several experts. Other provided physiological signals include multiple electrooculogram (EOG) and electromyography (EMG) channels, sampled at 200Hz."

    def get_cols(self, allcols):
        feature_cols_lookup = {
            "eeg": ["FP1-A1", "CZ-A1", "O1-A1", "FP2-A1", "O2-A1"],
            "eog": ["EOG1-A1", "EOG2-A1"],
            "all": allcols,
        }
        feature_cols = feature_cols_lookup[self.dataset_config.features]

        label_col_lookup = {
            "eeg": "EEG_label",
            "eog": "EOG_label",
            "all": "ALL_label",
        }
        label_col = label_col_lookup[self.dataset_config.labels]

        return feature_cols, label_col

    def get_data(self, split=None):
        split = split or self.split

        assert self.dataset_config.version == "v2"
        basepath = Path(__file__).parent / "../data/dreams/v2/"
        split_fn = "train.csv" if split == "train" else "test.csv"
        data = pd.read_csv(basepath / split_fn)

        time_col = "ts"
        clip_col = "patient_ID"
        allcols = data.columns.difference([time_col, clip_col])
        feature_cols, label_col = self.get_cols(allcols)

        xs = data[feature_cols].values
        clip_ids = data[clip_col].values.astype(int)
        timestamps = data[time_col].values

        if split == "train":
            labels = None
        else:
            labels_df = pd.read_csv(basepath / "test_label.csv")
            labels = labels_df[label_col].values.astype(int)
            assert labels_df[clip_col].equals(data[clip_col])
            assert labels_df[time_col].equals(data[time_col])

        desc_fn = "train_data_desc.csv" if split == "train" else "test_data_desc.csv"
        descriptions = pd.read_csv(basepath / desc_fn, index_col=0)
        descriptions = descriptions["data_desc"].to_dict()
        descriptions = {k: f"Patient description: {v}" for k, v in descriptions.items()}

        return {
            "data": xs,
            "labels": labels,
            "clip_ids": clip_ids,
            "clip_descriptions": descriptions,
            "timestamps": timestamps,
        }


class DreamsForecastDataset(DreamsDataset, ForecastDataset):
    pass

class DreamsReconstructionDataset(DreamsDataset, ReconstructionDataset):
    pass

class DreamsAnomalyDetectionDataset(DreamsDataset, AnomalyDetectionDataset):
    pass


dreams_datasets = {
    "forecasting": DreamsForecastDataset,
    "reconstruction": DreamsReconstructionDataset,
    "anomaly_detection": DreamsAnomalyDetectionDataset,
}
