from abc import ABC
from pathlib import Path
import pandas as pd
import re

from .base import BaseDataset, ForecastDataset, ReconstructionDataset, SemanticSegmentationDataset


class VentilatorDataset(BaseDataset, ABC):

    supported_tasks = [
        "forecasting",
        "reconstruction",
        "semantic_segmentation",
    ]

    description = "The dataset contains time-series data of airway pressure and flow rate measurements collected from a mechanical ventilator during the respiratory support of a fully sedated patient. The data is sampled at a frequency of 100 Hz. The airway pressure is measured in cmH2O and the flow rate is measured in L/min."


class VentilatorForecastingDataset(VentilatorDataset, ForecastDataset):

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


class VentilatorReconstructionDataset(VentilatorDataset, ReconstructionDataset):

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

    train_clips = [
        "patient_572_vent_w_1_labeled",
        "patient_572_vent_w_2_labeled",
        "patient_572_vent_w_4_labeled", # async
        "patient_674_vent_w_1_labeled",
        "patient_674_vent_w_4_labeled",
        "patient_697_vent_w_1_labeled",
        "patient_697_vent_w_2_labeled",
    ]
    test_clips = [
        "patient_704_vent_w_1_labeled", # anom
        "patient_709_vent_w_1_labeled",
        "patient_709_vent_w_2_labeled", # async
    ]

    @property
    def n_classes(self):
        return 2

    def get_data(self, split=None):
        split = split or self.split

        assert self.dataset_config.version == "v4"
        assert self.dataset_config.split_version == "v1"

        basepath = Path(__file__).parent / "../data/ventilator/v4/"
        clip_list = self.train_clips if split == "train" else self.test_clips

        dfs = []
        for clip_id in clip_list:
            fn = basepath / f"{clip_id}.csv"
            df = pd.read_csv(fn)
            df = df[df.label >= 0]
            df["clip_id"] = parse_clip_id(clip_id)
            dfs.append(df)
        data = pd.concat(dfs, ignore_index=True)

        features = data[["pressure", "flow"]].values
        labels = data["label"].values.astype(int)
        clip_ids = data["clip_id"].values
        timestamps = data["dt"].values.astype(float)

        return {"data": features, "labels": labels, "clip_ids": clip_ids, "timestamps": timestamps}


def parse_clip_id(clip_string):
    match = re.match(r"patient_(\d+)_vent_w_(\d+)", clip_string)
    patient_id, clip_number = match.groups()
    patient_id, clip_number = int(patient_id), int(clip_number)
    clip_id = (patient_id * 100) + clip_number
    print(f"Clip string: {clip_string}, Clip ID: {clip_id}")
    return clip_id


ventilator_datasets = {
    "forecasting": VentilatorForecastingDataset,
    "reconstruction": VentilatorReconstructionDataset,
    "semantic_segmentation": VentilatorSemanticSegmentationDataset,
}
