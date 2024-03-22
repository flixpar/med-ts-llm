from abc import ABC, abstractmethod
from pathlib import Path
import toml, json
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader

import pytorch_optimizer

import wandb

from models import model_lookup
from datasets import get_dataset
from loggers import get_logger
from utils import set_seed, dict_to_object

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.set_float32_matmul_precision("medium")


class BaseTask(ABC):

    def __init__(self, run_id, config, newrun=True):
        self.run_id = run_id
        self.config = config
        self.newrun = newrun

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
        self.best_score = float("inf")

        self.logger = get_logger(self, self.config, self.newrun)

    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def val(self):
        pass

    @abstractmethod
    def test(self):
        pass

    @abstractmethod
    def score(self, pred, target):
        pass

    def build_model(self):
        model_cls = model_lookup[self.config.model]
        self.model = model_cls(self.config, self.train_dataset)
        assert self.task in self.model.supported_tasks, f"{self.task} not supported by {self.config.model}"
        return self.model

    def build_optimizer(self):
        params = [p for p in self.model.parameters() if p.requires_grad]
        match self.config.training.optimizer:
            case "adam":
                self.optimizer = optim.Adam(params, lr=self.config.training.learning_rate)
            case "adamw":
                self.optimizer = optim.AdamW(params, lr=self.config.training.learning_rate, weight_decay=0.01)
            case "sgd":
                self.optimizer = optim.SGD(params, lr=self.config.training.learning_rate, momentum=0.9, nesterov=True)
            case "ranger21" | "ranger":
                num_iter = len(self.train_dataloader) * self.config.training.epochs
                self.optimizer = pytorch_optimizer.Ranger21(params, num_iterations=num_iter, lr=self.config.training.learning_rate)
            case _:
                raise ValueError(f"Invalid optimizer selection: {self.config.training.optimizer}")
        return self.optimizer

    def build_loss(self):
        match self.config.training.loss:
            case "mse":
                self.loss_fn = torch.nn.MSELoss()
            case "mae":
                self.loss_fn = torch.nn.L1Loss()
            case _:
                raise ValueError(f"Invalid loss function selection: {self.config.training.loss}")
        return self.loss_fn

    def build_datasets(self):
        self.train_dataset = get_dataset(self.config, "train")
        self.val_dataset = get_dataset(self.config, "val")
        self.test_dataset = get_dataset(self.config, "test")

    def build_dataloaders(self):
        num_workers = self.config.setup.num_workers
        num_workers = os.cpu_count()//2 if num_workers == "auto" else num_workers
        self.train_dataloader = DataLoader(
            self.train_dataset,
            batch_size = self.config.training.batch_size,
            shuffle = True,
            num_workers = num_workers,
        )
        self.val_dataloader = DataLoader(
            self.val_dataset,
            batch_size = self.config.training.batch_size,
            shuffle = False,
            num_workers = num_workers,
        )
        self.test_dataloader = DataLoader(
            self.test_dataset,
            batch_size = self.config.training.batch_size,
            shuffle = False,
            num_workers = num_workers,
        )

    def log_end(self):
        self.logger.log_end()

    def log_step(self, loss):
        self.step += self.config.training.batch_size
        self.logger.log_scores(train_loss=loss)

    def log_epoch(self, scores={}, **kwscores):
        scores = scores | kwscores
        self.logger.log_scores(scores)
        self.logger.save_state("latest")

        metric = "val_" + self.config.training.eval_metric
        if scores[metric] < self.best_score:
            self.best_score = scores[metric]
            self.logger.save_state("best")

        self.epoch += 1

    def log_scores(self, scores={}, **kwscores):
        self.logger.log_scores(scores | kwscores)

    def get_device(self):
        match self.config.setup.device, torch.cuda.is_available():
            case "auto", True:
                return torch.device("cuda")
            case "auto", False:
                return torch.device("cpu")
            case x, _:
                return torch.device(x)

    def get_dtype(self):
        match self.config.setup.dtype:
            case "bfloat16" | "bf16":
                return torch.bfloat16
            case "float16" | "half" | "fp16" | "16" | 16:
                return torch.float16
            case "float32" | "float" | "fp32" | "32" | 32:
                return torch.float32
            case "mixed":
                return torch.float32
            case _:
                raise ValueError(f"Invalid dtype selection: {self.config.setup.dtype}")

    @classmethod
    def from_run_id(cls, run_id):
        basepath = Path(__file__).parent / f"../outputs/logs/{run_id}/"
        config = toml.load(basepath / "config.toml")
        config = dict_to_object(config)

        trainer = cls(run_id, config, newrun=False)

        modelpath = basepath / "checkpoints/latest.pt"
        state = torch.load(modelpath)
        _, unexpected = trainer.model.load_state_dict(state["model"], strict=False)
        assert not unexpected, f"Unexpected keys in model state: {unexpected}"

        trainer.epoch = state["epoch"]
        trainer.step = state["step"]

        return trainer
