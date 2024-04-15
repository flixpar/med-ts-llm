from abc import ABC
from pathlib import Path

import numpy as np
import pandas as pd

from .base import (
    BaseDataset,
    ClipDataset,
    ForecastDataset,
    ReconstructionDataset,
    SemanticSegmentationDataset,
)


class LUDBDataset(BaseDataset, ABC):

    supported_tasks = ["forecasting", "reconstruction", "semantic_segmentation"]
    description = "LUDB is an ECG signal database collected from subjects with various cardiovascular diseases used for ECG delineation. Cardiologists manually annotated boundaries of P, T waves and QRS complexes. Each clip consists of a 10 second signal from a single ECG lead, sampled at 500Hz."

    def get_data(self, split=None):
        split = split or self.split

        assert self.dataset_config.version == "v3"
        basepath = Path(__file__).parent / "../data/ludb/"
        split_fn = "train.csv" if split == "train" else "test.csv"
        data = pd.read_csv(basepath / split_fn)

        data.lead = data.lead.astype("category")
        data["lead_idx"], lead_cats = data.lead.factorize()

        patient_ids = data.patient_id.values.astype(int)
        lead_ids = data.lead_idx.values.astype(int)
        data["clip_id"] = (patient_ids * 100) + lead_ids

        data.time = data.time.str.slice(start=14).astype(float)
        data = data.sort_values(["clip_id", "time"]).reset_index(drop=True)

        features = data.ecg.values[:,np.newaxis]
        labels = data.label.values.astype(int)
        timestamps = data.time.values
        clip_ids = data.clip_id.values.astype(int)

        lead_descriptions = dict(enumerate(lead_cats))
        lead_descriptions = {k: f"ECG lead: {v}" for k, v in lead_descriptions.items()}

        desc_fn = "train_data_desc_cleaned.csv" if split == "train" else "test_data_desc_cleaned.csv"
        patient_descriptions = pd.read_csv(basepath / desc_fn, index_col=0)
        patient_descriptions = patient_descriptions["data_desc"].to_dict()
        patient_descriptions = {k: f"Patient information: {v}" for k, v in patient_descriptions.items()}

        descriptions = {(p*100)+l: dp + "; " + dl for (p,dp) in patient_descriptions.items() for (l,dl) in lead_descriptions.items()}

        return {
            "data": features,
            "labels": labels,
            "timestamps": timestamps,
            "clip_ids": clip_ids,
            "clip_descriptions": descriptions,
        }


class LUDBForecastingDataset(LUDBDataset, ForecastDataset):
    pass


class LUDBReconstructionDataset(LUDBDataset, ReconstructionDataset):
    pass


class LUDBSemanticSegmentationDataset(LUDBDataset, ClipDataset, SemanticSegmentationDataset):
    n_classes = 4
    task_description = "Segment the following ECG signal into P waves, T waves, and QRS complexes."


ludb_datasets = {
    "forecasting": LUDBForecastingDataset,
    "reconstruction": LUDBReconstructionDataset,
    "semantic_segmentation": LUDBSemanticSegmentationDataset,
}
