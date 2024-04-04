from abc import ABC
from pathlib import Path
import pandas as pd

from .base import BaseDataset, ForecastDataset, SemanticSegmentationDataset


class VentilatorDataset(BaseDataset, ABC):

    supported_tasks = [
        "forecasting",
        "semantic_segmentation",
    ]

    description = "The dataset contains time-series data of airway pressure and flow rate measurements collected from a mechanical ventilator during the respiratory support of a fully sedated patient. The data is sampled at a frequency of 100 Hz. The airway pressure is measured in cmH2O and the flow rate is measured in L/min."


class VentilatorForecastingDataset(VentilatorDataset, ForecastDataset):

    def __init__(self, config, split):
        super().__init__(config, split)

    def get_data(self, split=None):
        split = split or self.split

        basepath = Path(__file__).parent / "../data/ventilator/v1/"
        waveform_files = sorted(basepath.glob("*.csv"))

        dfs = [pd.read_csv(f, usecols=["pressure", "flow"]) for f in waveform_files]
        data = pd.concat(dfs, ignore_index=True).values

        train_pct, val_pct, test_pct = 0.7, 0.15, 0.15
        train_idx = int(train_pct * data.shape[0])
        val_idx = int((train_pct + val_pct) * data.shape[0])

        match split:
            case "train":
                data = data[:train_idx,:]
            case "val":
                data = data[train_idx:val_idx,:]
            case "test":
                data = data[val_idx:,:]
            case _:
                raise ValueError(f"Invalid split: {split}")

        return {"data": data}


class VentilatorSemanticSegmentationDataset(VentilatorDataset, SemanticSegmentationDataset):

    def __init__(self, config, split):
        super().__init__(config, split)

    @property
    def n_classes(self):
        return 2

    def get_data(self, split=None):
        split = split or self.split

        basepath = Path(__file__).parent / "../data/ventilator/v3/"
        waveform_files = sorted(basepath.glob("*.csv"))

        dfs = [pd.read_csv(f, usecols=["pressure", "flow", "label"]) for f in waveform_files]
        dfs = [df[df.label >= 0] for df in dfs]
        dfs = [df for df in dfs if len(df) > 1000]
        data = pd.concat(dfs, ignore_index=True)

        labels = data["label"].values.astype(int)
        data = data[["pressure", "flow"]].values

        train_pct, val_pct, test_pct = 0.7, 0.15, 0.15
        train_idx = int(train_pct * data.shape[0])
        val_idx = int((train_pct + val_pct) * data.shape[0])

        match split:
            case "train":
                data = data[:train_idx,:]
                labels = labels[:train_idx]
            case "val":
                data = data[train_idx:val_idx,:]
                labels = labels[train_idx:val_idx]
            case "test":
                data = data[val_idx:,:]
                labels = labels[val_idx:]
            case _:
                raise ValueError(f"Invalid split: {split}")

        return {"data": data, "labels": labels}


ventilator_datasets = {
    "forecasting": VentilatorForecastingDataset,
    "semantic_segmentation": VentilatorSemanticSegmentationDataset,
}
