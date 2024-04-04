from abc import ABC
from pathlib import Path
import pandas as pd

from .base import BaseDataset, ForecastDataset, SegmentationDataset


class BIDMCDataset(BaseDataset, ABC):

    supported_tasks = ["forecasting", "segmentation"]

    description = "The BIDMC dataset is a dataset of electrocardiogram (ECG), pulse oximetry (photoplethysmogram, PPG) and impedance pneumography respiratory signals acquired from intensive care patients. Two annotators manually annotated individual breaths in each recording using the impedance respiratory signal."

    def get_data(self, split=None):
        split = split or self.split

        basepath = Path(__file__).parent / "../data/bidmc/"
        split_fn = "train.csv" if split == "train" else "test.csv"
        data = pd.read_csv(basepath / split_fn)

        time_col = "Time"
        clip_col = "patient_id"
        label_col = "label"
        feature_cols = data.columns.difference([time_col, clip_col, label_col])

        xs = data[feature_cols].values
        labels = data[label_col].values.astype(int)
        clip_ids = data[clip_col].values.astype(int)
        timestamps = data[time_col].values

        return {"data": xs, "labels": labels, "clip_ids": clip_ids, "timestamps": timestamps}


class BIDMCForecastingDataset(BIDMCDataset, ForecastDataset):
    pass

class BIDMCSegmentationDataset(BIDMCDataset, SegmentationDataset):
    pass


bidmc_datasets = {
    "forecasting": BIDMCForecastingDataset,
    "segmentation": BIDMCSegmentationDataset,
}
