from abc import ABC
from pathlib import Path
import pandas as pd

from .base import BaseDataset, ForecastDataset


class ETTDataset(BaseDataset, ABC):

    supported_tasks = ["forecasting"]
    description = "The Electricity Transformer Temperature (ETT) is a crucial indicator in the electric power long-term deployment."

    def get_data(self, split=None):
        split = split or self.split

        basepath = Path(__file__).parent / "../data/ett/"
        datapath = basepath / (self.config.data.dataset + ".csv")
        data = pd.read_csv(datapath, parse_dates=["date"], index_col="date")

        train_range = (0, 12 * 30 * 24)
        val_range = (train_range[1], train_range[1] + 4 * 30 * 24)
        test_range = (val_range[1], val_range[1] + 4 * 30 * 24)

        match split:
            case "train":
                data = data.iloc[slice(*train_range)].values
            case "val":
                data = data.iloc[slice(*val_range)].values
            case "test":
                data = data.iloc[slice(*test_range)].values
            case _:
                raise ValueError(f"Invalid split: {split}")

        return {"data": data}


class ETTForecastDataset(ETTDataset, ForecastDataset):
    pass


ett_datasets = {
    "forecasting": ETTForecastDataset,
}
