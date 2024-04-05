from .reconstruction import ReconstructionTask
from datasets import PretrainingDataset, get_dataset


class PretrainingTask(ReconstructionTask):

    def __init__(self, run_id, config, newrun=True):
        super().__init__(run_id, config, newrun)
        self.task = "pretraining"

    def build_datasets(self):
        train_datasets, val_datasets, test_datasets = {}, {}, {}
        datasets = ["ECG", "ventilator", "bidmc", "ludb"]
        for dataset_name in datasets:
            cfg = self.config.copy()
            cfg.data.dataset = dataset_name
            cfg.task = "reconstruction"
            train_datasets[dataset_name] = get_dataset(cfg, "train")
            val_datasets[dataset_name] = get_dataset(cfg, "val")
            test_datasets[dataset_name] = get_dataset(cfg, "test")

        downsample_pct = self.config.tasks.pretraining.downsample_pct
        n_features = self.config.tasks.pretraining.n_features
        self.train_dataset = PretrainingDataset(train_datasets, downsample_pct=downsample_pct, n_features=n_features)
        self.val_dataset = PretrainingDataset(val_datasets, downsample_pct=downsample_pct, n_features=n_features)
        self.test_dataset = PretrainingDataset(test_datasets, downsample_pct=downsample_pct, n_features=n_features)
