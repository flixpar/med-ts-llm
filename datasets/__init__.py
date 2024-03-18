from .ett import ETTDataset

from .util import Multi2UniDataset


dataset_lookup = {
    "ETTh1": ETTDataset,
    "ETTh2": ETTDataset,
    "ETTm1": ETTDataset,
    "ETTm2": ETTDataset,
}

def get_dataset(config, split):
    dataset_cls = dataset_lookup[config.data.dataset]
    dataset = dataset_cls(config, split)
    if config.data.mode == "univariate" and dataset.mode == "multivariate":
        dataset = Multi2UniDataset(dataset)
    return dataset
