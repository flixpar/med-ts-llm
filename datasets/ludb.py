from abc import ABC
from pathlib import Path

import numpy as np
import pandas as pd

from .base import BaseDataset, ForecastDataset, ReconstructionDataset, SemanticSegmentationDataset


class LUDBDataset(BaseDataset, ABC):

    supported_tasks = ["forecasting", "reconstruction", "semantic_segmentation"]
    description = "LUDB is an ECG signal database with marked boundaries and peaks of P, T waves and QRS complexes. Cardiologists manually annotated all 200 records of healthy and sick patients which contains a corresponding diagnosis. This can be used for ECG delineation."

    def get_data(self, split=None):
        split = split or self.split

        basepath = Path(__file__).parent / "../data/ludb/"
        split_fn = "train.csv" if split == "train" else "test.csv"
        data = pd.read_csv(basepath / split_fn)

        desc_fn = "train_data_desc.csv" if split == "train" else "test_data_desc.csv"
        descriptions = pd.read_csv(basepath / desc_fn, index_col=0)
        descriptions = descriptions["data_desc"].to_dict()

        features = data["ecg"].values[:,np.newaxis]
        labels = data["label"].values.astype(int)
        clip_ids = data["patient_id"].values.astype(int)

        return {"data": features, "labels": labels, "clip_ids": clip_ids, "clip_descriptions": descriptions}



class LUDBForecastingDataset(LUDBDataset, ForecastDataset):
    pass


class LUDBReconstructionDataset(LUDBDataset, ReconstructionDataset):
    pass


class LUDBSemanticSegmentationDataset(LUDBDataset, SemanticSegmentationDataset):
    def __getitem__(self, idx):
        idx = idx * self.step_size
        idx_range = (idx, idx + self.pred_len)

        x = self.data[slice(*idx_range),:]
        y = self.labels[slice(*idx_range)]

        patient_id = self.patient_ids[idx]
        desc = self.descriptions[patient_id]

        return {"x_enc": x, "labels": y, "descriptions": f"Patient description: {desc}"}


ludb_datasets = {
    "forecasting": LUDBForecastingDataset,
    "reconstruction": LUDBReconstructionDataset,
    "semantic_segmentation": LUDBSemanticSegmentationDataset,
}
