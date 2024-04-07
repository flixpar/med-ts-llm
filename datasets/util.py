import math
import bisect

import numpy as np

import torch
from torch.utils.data import Dataset


def multi_2_uni_dataset(dataset_cls):
    class Multi2UniDataset(dataset_cls):
        univariate = True

        @property
        def n_features(self):
            return 1

        @property
        def real_features(self):
            return super().n_features

        def __getitem__(self, index):
            example_idx = index // self.real_features
            feature_idx = index % self.real_features
            inputs = super().__getitem__(example_idx)

            inputs["x_enc"] = inputs["x_enc"][:, feature_idx:(feature_idx+1)]
            if "y" in inputs:
                inputs["y"] = inputs["y"][:, feature_idx:(feature_idx+1)]
            if "x_dec" in inputs:
                inputs["x_dec"] = inputs["x_dec"][:, feature_idx:(feature_idx+1)]

            return inputs

        def __len__(self):
            return super().__len__() * self.real_features

        def inverse_index(self, index):
            example_idx = super().inverse_index(index // self.real_features)
            feature_idx = index % self.real_features
            return example_idx, feature_idx

    return Multi2UniDataset


class PretrainingDataset(Dataset):

    supported_tasks = ["pretraining"]
    description = "This dataset consists of a mix of different biomedical time series datasets."

    def __init__(self, datasets, downsample_pct=1.0, n_features=None):
        self.datasets = list(datasets.values())
        self.dataset_names = list(datasets.keys())

        self.config = self.datasets[0].config
        self.split = self.datasets[0].split
        self.task = "pretraining"
        self.name = "pretrain:" + "+".join(self.dataset_names)

        self.task_config = self.config.get("tasks", {}).get(self.task, {})
        self.dataset_config = {}
        self.data_config = self.config.data

        inds_subset = lambda dataset: torch.randperm(len(dataset))[:max(1, int(downsample_pct * len(dataset)))]
        self.dataset_inds = [inds_subset(dataset) for dataset in self.datasets]

        self.lens = [len(inds) for inds in self.dataset_inds]
        self.cumsums = [sum(self.lens[:i]) for i in range(len(self.datasets))]

        if n_features is None or n_features == "auto":
            n_features = max(dataset.n_features for dataset in self.datasets)
        self.n_features = n_features
        self.real_features = n_features
        self.n_classes = 0

        self.univariate = False
        self.clip_dataset = False

        self.pred_len = self.datasets[0].pred_len
        self.history_len = self.datasets[0].history_len
        self.step_size = self.datasets[0].step_size
        self.n_points = sum(self.step_size * l for l in self.lens)

    def __getitem__(self, index):
        dataset_idx = bisect.bisect_right(self.cumsums, index) - 1
        index_in_dataset = index - self.cumsums[dataset_idx]
        index_in_dataset = self.dataset_inds[dataset_idx][index_in_dataset]

        item = self.datasets[dataset_idx][index_in_dataset]
        item["x_enc"] = self.adjust_n_features(item["x_enc"])
        if "y" in item:
            item["y"] = self.adjust_n_features(item["y"])

        meta = {"dataset": self.dataset_names[dataset_idx], "dataset_description": self.datasets[dataset_idx].description}
        return item | meta

    def __len__(self):
        return sum(self.lens)

    def adjust_n_features(self, x):
        if x.shape[1] < self.n_features:
            repeats = math.ceil(self.n_features / x.shape[1])
            x = x.repeat(1, repeats)
        if x.shape[1] > self.n_features:
            x = x[:, :self.n_features]
        return x

    def inverse_index_full(self, index):
        dataset_idx = bisect.bisect_right(self.cumsums, index) - 1
        index_in_dataset = index - self.cumsums[dataset_idx]
        index_in_dataset = self.dataset_inds[dataset_idx][index_in_dataset]
        index_in_dataset = self.datasets[dataset_idx].inverse_index(index_in_dataset)
        return dataset_idx, index_in_dataset

    def inverse_index(self, idx):
        idx = idx * self.step_size
        x_range = (idx, idx + self.pred_len)
        return x_range
