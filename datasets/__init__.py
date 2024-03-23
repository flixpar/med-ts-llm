from .ett import ETTDataset
from .psm import PSMDatasetSelector
from .msl import MSLDatasetSelector
from .ecg import ECGMITDatasetSelector
from .ventilator import VentilatorDatasetSelector
from .bidmc import BIDMCDatasetSelector
from .ludb import LUDBDatasetSelector

from .util import Multi2UniDataset
from models import model_lookup


dataset_lookup = {
    "ETTh1": ETTDataset,
    "ETTh2": ETTDataset,
    "ETTm1": ETTDataset,
    "ETTm2": ETTDataset,
    "PSM": PSMDatasetSelector,
    "MSL": MSLDatasetSelector,
    "ECG": ECGMITDatasetSelector,
    "ventilator": VentilatorDatasetSelector,
    "bidmc": BIDMCDatasetSelector,
    "ludb": LUDBDatasetSelector,
}

def get_dataset(config, split):
    dataset_cls = dataset_lookup[config.data.dataset]
    dataset = dataset_cls(config, split)

    if not config.task in dataset.supported_tasks:
        raise ValueError(f"Task {config.task} not supported by dataset {config.data.dataset}")

    assert config.data.mode == "multivariate"

    model_cls = model_lookup[config.model]
    if dataset.mode == "multivariate" and "multivariate" not in model_cls.supported_modes:
        dataset = Multi2UniDataset(dataset)

    return dataset
