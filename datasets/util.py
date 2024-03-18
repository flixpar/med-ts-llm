from torch.utils.data import Dataset


class Multi2UniDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        self.n_features = dataset.n_features
        self.__dict__.update(dataset.__dict__)

    def __getitem__(self, index):
        example_idx = index // self.n_features
        feature_idx = index % self.n_features

        x_enc, x_dec, y = self.dataset[example_idx]
        x_enc, x_dec, y = x_enc[:, feature_idx], x_dec[:, feature_idx], y[:, feature_idx]

        return x_enc, x_dec, y

    def __len__(self):
        return len(self.dataset) * self.n_features
