from .ett import ett_datasets
from .psm import psm_datasets
from .msl import msl_datasets
from .ecg import ecg_datasets
from .ventilator import ventilator_datasets
from .bidmc import bidmc_datasets
from .ludb import ludb_datasets

from .util import multi_2_uni_dataset
from .util import PretrainingDataset


dataset_lookup = {
    "ETTh1": ett_datasets,
    "ETTh2": ett_datasets,
    "ETTm1": ett_datasets,
    "ETTm2": ett_datasets,
    "PSM": psm_datasets,
    "MSL": msl_datasets,
    "ECG": ecg_datasets,
    "ventilator": ventilator_datasets,
    "bidmc": bidmc_datasets,
    "ludb": ludb_datasets,
}

def get_dataset(config, split):
    dataset_cls = dataset_lookup[config.data.dataset][config.task]

    if config.data.mode == "univariate" and dataset.mode == "multivariate":
        dataset = multi_2_uni_dataset(dataset)

    if not config.task in dataset_cls.supported_tasks:
        raise ValueError(f"Task {config.task} not supported by dataset {config.data.dataset}")

    dataset = dataset_cls(config, split)
    return dataset
