from pathlib import Path

import pandas as pd

from torch.utils.data import Dataset


class ETTDataset(Dataset):
    def __init__(self, config, split):
        assert config.data.cols == "all"
        assert config.data.normalize == False
        assert config.data.step == 1

        self.split = split
        self.history_len = config.history_len
        self.pred_len = config.pred_len

        basepath = Path(__file__).parent / "../data/ett/"
        datapath = basepath / (config.data.dataset + ".csv")
        data = pd.read_csv(datapath, parse_dates=["date"], index_col="date")

        train_range = (0, 12 * 30 * 24)
        val_range = (train_range[1], train_range[1] + 4 * 30 * 24)
        test_range = (val_range[1], val_range[1] + 4 * 30 * 24)

        train = data.iloc[slice(*train_range)].values
        val = data.iloc[slice(*val_range)].values
        test = data.iloc[slice(*test_range)].values

        match split:
            case "train":
                self.data = train
            case "val":
                self.data = val
            case "test":
                self.data = test
            case _:
                raise ValueError(f"Invalid split: {split}")

        if config.data.normalize:
            pass

        self.n_features = self.data.shape[1]
        self.mode = "multivariate"

        self.description = "The Electricity Transformer Temperature (ETT) is a crucial indicator in the electric power long-term deployment."

    def __len__(self):
        return self.data.shape[0] - self.history_len - self.pred_len + 1

    def __getitem__(self, idx):
        x_range = (idx, idx + self.history_len)
        y_range = (x_range[1], x_range[1] + self.pred_len)

        x = self.data[slice(*x_range),:]
        y = self.data[slice(*y_range),:]

        x_dec = x[0:0,:]

        return x, x_dec, y
