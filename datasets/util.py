from torch.utils.data import Dataset


class Multi2UniDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        self.__dict__.update(dataset.__dict__)
        self.real_features = dataset.n_features
        self.n_features = 1

    def __getitem__(self, index):
        example_idx = index // self.real_features
        feature_idx = index % self.real_features

        if self.task == "forecasting":
            inputs = self.dataset[example_idx]
            inputs["x_enc"] = inputs["x_enc"][:, feature_idx]
            inputs["y"] = inputs["y"][:, feature_idx]
            if "x_dec" in inputs:
                inputs["x_dec"] = inputs["x_dec"][:, feature_idx]
            return inputs
        elif self.task in ["anomaly_detection", "semantic_segmentation", "segmentation"]:
            inputs = self.dataset[example_idx]
            inputs["x_enc"] = inputs["x_enc"][:, feature_idx]
            if "x_dec" in inputs:
                inputs["x_dec"] = inputs["x_dec"][:, feature_idx]
            return inputs
        else:
            raise ValueError(f"Task {self.task} not supported by Multi2UniDataset")

    def __len__(self):
        return len(self.dataset) * self.real_features

    def inverse_index(self, index):
        example_idx = self.dataset.inverse_index(index // self.real_features)
        feature_idx = index % self.real_features
        return example_idx, feature_idx

