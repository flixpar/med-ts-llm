from abc import ABC
from pathlib import Path
import numpy as np

from .base import BaseDataset, ForecastDataset, AnomalyDetectionDataset


class MSLDataset(BaseDataset, ABC):

    supported_tasks = ["forecasting", "anomaly_detection"]
    description = "The MSL (Mars Science Laboratory rover) dataset was created by NASA and consists of telemetry data across 55 sensors on the rover. The data is collected at 1 minute intervals and spans a period of 78 Martian days. The dataset is labeled with 143 anomalous intervals, each of which is labeled by an expert as an incident, surprise, or an anomaly."

    def get_data(self, split=None):
        split = split or self.split
        basepath = Path(__file__).parent / "../data/msl/"
        split_fn = "MSL_train.npy" if split == "train" else "MSL_test.npy"
        data = np.load(basepath / split_fn)
        return {"data": data}


class MSLForecastingDataset(MSLDataset, ForecastDataset):
    pass


class MSLAnomalyDetectionDataset(MSLDataset, AnomalyDetectionDataset):
    def get_data(self, split=None):
        split = split or self.split
        data = super().get_data(split)

        if self.split != "train":
            basepath = Path(__file__).parent / "../data/msl/"
            labels = np.load(basepath / "MSL_test_label.npy")
            labels = labels.astype(int)
        else:
            labels = None

        return data | {"labels": labels}


msl_datasets = {
    "forecasting": MSLForecastingDataset,
    "anomaly_detection": MSLAnomalyDetectionDataset,
}
