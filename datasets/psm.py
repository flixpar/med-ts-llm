from abc import ABC
from pathlib import Path

import pandas as pd
import numpy as np

from .base import BaseDataset, ForecastDataset, AnomalyDetectionDataset


class PSMDataset(BaseDataset, ABC):

    supported_tasks = ["forecasting", "anomaly_detection"]
    description = "The PSM dataset is proposed by eBay and consists of 26 dimensional data captured internally from application server nodes. The dataset is used to predict the number of sessions in the next 10 minutes based on the current and historical data."

    def get_data(self, split=None):
        split = split or self.split
        basepath = Path(__file__).parent / "../data/psm/"
        split_fn = "train.csv" if split == "train" else "test.csv"
        data = pd.read_csv(basepath / split_fn)
        data = data.drop(columns=["timestamp_(min)"])
        data = np.nan_to_num(data.values)
        return {"data": data}


class PSMForecastingDataset(PSMDataset, ForecastDataset):
    pass


class PSMAnomalyDetectionDataset(PSMDataset, AnomalyDetectionDataset):
    def get_data(self, split=None):
        split = split or self.split
        data = super().get_data(split)

        if self.split != "train":
            basepath = Path(__file__).parent / "../data/psm/"
            labels = pd.read_csv(basepath / "test_label.csv")
            labels = labels.drop(columns=["timestamp_(min)"])
            labels = labels.values[:,0].astype(int)
        else:
            labels = None

        return data | {"labels": labels}


psm_datasets = {
    "forecasting": PSMForecastingDataset,
    "anomaly_detection": PSMAnomalyDetectionDataset,
}
