from abc import ABC, abstractmethod
from pathlib import Path
import toml, json

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

from torch.utils.data import DataLoader

import wandb

from models import model_lookup
from datasets import get_dataset
from utils import set_seed


class BaseTask(ABC):

    def __init__(self, run_id, config):
        self.run_id = run_id
        self.config = config

        self.device = self.get_device()
        self.dtype = self.get_dtype()

        set_seed(self.config.setup.seed)

        self.build_datasets()
        self.build_dataloaders()

        self.model = self.build_model().to(self.device, self.dtype)
        self.optimizer = self.build_optimizer()
        self.loss_fn = self.build_loss()

        self.epoch = 1
        self.step = 0

        self.log_start()

    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def val(self):
        pass

    @abstractmethod
    def test(self):
        pass

    def build_model(self):
        model_cls = model_lookup[self.config.model]
        self.model = model_cls(self.config, self.train_dataset)
        return self.model

    def build_optimizer(self):
        match self.config.training.optimizer:
            case "adam":
                self.optimizer = optim.Adam(self.model.parameters(), lr=self.config.training.learning_rate)
            case _:
                raise ValueError(f"Invalid optimizer selection: {self.config.optimizer}")
        return self.optimizer

    def build_loss(self):
        match self.config.training.loss:
            case "mse":
                self.loss_fn = torch.nn.MSELoss()
            case "mae":
                self.loss_fn = torch.nn.L1Loss()
            case _:
                raise ValueError(f"Invalid loss function selection: {self.config.loss}")
        return self.loss_fn

    def build_datasets(self):
        self.train_dataset = get_dataset(self.config, "train")
        self.val_dataset = get_dataset(self.config, "val")
        self.test_dataset = get_dataset(self.config, "test")

    def build_dataloaders(self):
        self.train_dataloader = DataLoader(
            self.train_dataset,
            batch_size = self.config.training.batch_size,
            shuffle = True,
            num_workers = self.config.setup.num_workers,
        )
        self.val_dataloader = DataLoader(
            self.val_dataset,
            batch_size = self.config.training.batch_size,
            shuffle = False,
            num_workers = self.config.setup.num_workers,
        )
        self.test_dataloader = DataLoader(
            self.test_dataset,
            batch_size = self.config.training.batch_size,
            shuffle = False,
            num_workers = self.config.setup.num_workers,
        )

    def log_start(self):
        self.logdir = Path(__file__).parent / f"../outputs/logs/{self.run_id}/"
        self.logdir.mkdir(parents=True)

        config_dict = self.config.to_dict()
        with open(self.logdir / "config.toml", "w") as f:
            toml.dump(config_dict, f)
        with open(self.logdir / "config.json", "w") as f:
            json.dump(config_dict, f, indent="\t")

        wandb.init(
            project = "med-time-llm",
            name = self.run_id,
            id = self.run_id,
            dir = self.logdir,
            config = self.config.__dict__,
            mode = "online" if not self.config.DEBUG else "disabled",
        )

    def log_end(self):
        wandb.finish()

    def log_step(self, loss):
        self.step += 1
        wandb.log({
            "epoch": self.epoch,
            "step": self.step,
            "train_loss": loss,
        })

    def log_epoch(self, scores={}, **kwscores):
        wandb.log({"epoch": self.epoch, "step": self.step} | scores | kwscores)
        self.epoch += 1

    def get_device(self):
        return torch.device(self.config.setup.device)

    def get_dtype(self):
        return torch.float32 if self.config.setup.device == "cpu" else torch.bfloat16
