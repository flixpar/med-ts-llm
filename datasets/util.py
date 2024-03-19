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
            x_enc, x_dec, y = self.dataset[example_idx]
            x_enc, y = x_enc[:, feature_idx], y[:, feature_idx]
            return x_enc, x_dec, y
        elif self.task == "anomaly_detection":
            x_enc, x_dec, label = self.dataset[example_idx]
            x_enc = x_enc[:, feature_idx]
            return x_enc, x_dec, label

    def __len__(self):
        return len(self.dataset) * self.real_features

    def inverse_index(self, index):
        example_idx = self.dataset.inverse_index(index // self.real_features)
        feature_idx = index % self.real_features
        return example_idx, feature_idx

