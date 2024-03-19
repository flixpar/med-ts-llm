from .ett import ETTDataset
from .psm import PSMDatasetSelector

from .util import Multi2UniDataset


dataset_lookup = {
    "ETTh1": ETTDataset,
    "ETTh2": ETTDataset,
    "ETTm1": ETTDataset,
    "ETTm2": ETTDataset,
    "PSM": PSMDatasetSelector,
}

def get_dataset(config, split):
    dataset_cls = dataset_lookup[config.data.dataset]
    dataset = dataset_cls(config, split)

    if not config.task in dataset.supported_tasks:
        raise ValueError(f"Task {config.task} not supported by dataset {config.data.dataset}")

    if config.data.mode == "univariate" and dataset.mode == "multivariate":
        dataset = Multi2UniDataset(dataset)

    return dataset
