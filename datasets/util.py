from torch.utils.data import Dataset


class Multi2UniDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        self.n_features = 1
        self.__dict__.update(dataset.__dict__)

    def __getitem__(self, index):
        example_idx = index // self.n_features
        feature_idx = index % self.n_features

        if self.task == "forecasting":
            x_enc, x_dec, y = self.dataset[example_idx]
            x_enc, y = x_enc[:, feature_idx], y[:, feature_idx]
            return x_enc, x_dec, y
        elif self.task == "anomaly_detection":
            x_enc, x_dec, label = self.dataset[example_idx]
            x_enc = x_enc[:, feature_idx]
            return x_enc, x_dec, label

    def __len__(self):
        return len(self.dataset) * self.n_features
    
    def inverse_index(self, index):
        example_idx = index // self.n_features
        feature_idx = index % self.n_features
        return example_idx, feature_idx

