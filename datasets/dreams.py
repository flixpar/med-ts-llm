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
    description = "This data consists of digital 32-channel polysomnographic recordings (PSG), acquired from patients with different pathologies, in a sleep hospital laboratory. Muscle or movement artifacts on the electroencephalogram (EEG) were annotated in microevents or in sleep stages by several experts. Other provided physiological signals include multiple electrooculogram (EOG) and electromyography (EMG) channels, sampled at 200Hz."

    def get_data(self, split=None):
        split = split or self.split

        basepath = Path(__file__).parent / "../data/dreams/"
        split_fn = "train.csv" if split == "train" else "test.csv"
        data = pd.read_csv(basepath / split_fn)

        time_col = "ts"
        clip_col = "patient_ID"
        feature_cols = data.columns.difference([time_col, clip_col])

        data.sort_values([clip_col, time_col], inplace=True)
        data.reset_index(drop=True, inplace=True)

        xs = data[feature_cols].values
        clip_ids = data[clip_col].values.astype(int)
        timestamps = data[time_col].values

        if split == "train":
            labels = None
        else:
            labels_df = pd.read_csv(basepath / "test_label.csv")
            labels_df.sort_values([clip_col, time_col], inplace=True)
            labels_df.reset_index(drop=True, inplace=True)
            labels = labels_df["label"].values.astype(int)

            assert labels_df.patient_ID.equals(data.patient_ID)
            assert labels_df.ts.equals(data.ts)

        return {"data": xs, "labels": labels, "clip_ids": clip_ids, "timestamps": timestamps}


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
