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
        self.task = config.task

        self.device = self.get_device()
        self.dtype = self.get_dtype()

        set_seed(self.config.setup.seed)

        self.build_datasets()
        self.build_dataloaders()

        self.model = self.build_model().to(self.device, self.dtype)
        self.load_pretrained()

        self.optimizer = self.build_optimizer()
        self.scheduler = self.build_scheduler()
        self.loss_fn = self.build_loss().to(device=self.device)

        self.epoch = 1
        self.step = 0

        metric_dir = self.config.training.eval_metric_direction
        self.best_score = float("inf") if (metric_dir == "min") else float("-inf")

        self.logger = get_logger(self, self.config, self.newrun)
        signal.signal(signal.SIGUSR1, self.handle_termination)

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

    @abstractmethod
    def build_loss(self):
        pass

    @abstractmethod
    def predict(self, dataloader):
        pass

    def build_model(self):
        model_cls = model_lookup[self.config.model]
        self.model = model_cls(self.config, self.train_dataset)
        assert self.task in self.model.supported_tasks, f"{self.task} not supported by {self.config.model}"
        return self.model

    def build_optimizer(self):
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]

        if ("finetuning" in self.config) and self.config.finetuning.enabled and (self.config.finetuning.frozen_epochs > 0):
            pretrained_params = [p for (n, p) in self.model.named_parameters() if n in self.loaded_params]
            for p in pretrained_params:
                p.requires_grad = True
            params = [{"params": trainable_params}, {"params": pretrained_params, "lr": 0.0}]
        else:
            params = trainable_params

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

    def build_scheduler(self):
        scheduler_type = self.config.training.get("lr_scheduler")
        match scheduler_type:
            case None | "none" | "constant":
                self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=1, gamma=1)
            case _:
                raise ValueError(f"Invalid scheduler selection: {scheduler_type}")

        if ("finetuning" in self.config) and self.config.finetuning.enabled and (self.config.finetuning.frozen_epochs > 0):
            assert scheduler_type in ["none", "constant", None], "Frozen epochs only supported with constant scheduler"
            self.scheduler = FineTuningScheduler(
                self.optimizer,
                initial_lr=self.config.training.learning_rate,
                frozen_epochs=self.config.finetuning.frozen_epochs,
            )

        return self.scheduler

    def load_pretrained(self):
        if ("finetuning" not in self.config) or (not self.config.finetuning.enabled):
            return

        assert self.config.model == "timellm", "Only TimeLLM supports finetuning"

        cfg = self.config.finetuning

        pretrained_path = Path(__file__).parent / f"../outputs/logs/{cfg.pretrained_id}/checkpoints/{cfg.pretrained_ckpt}.pt"
        saved_state = torch.load(pretrained_path)["model"]
        loaded_params = self.model.load_pretrained(saved_state)

        if cfg.freeze_forever:
            for name, param in self.model.named_parameters():
                if name in loaded_params:
                    param.requires_grad = False

        if cfg.frozen_epochs > 0:
            assert not cfg.freeze_forever, "Cannot freeze forever and then unfreeze"
            self.loaded_params = loaded_params
            for name, param in self.model.named_parameters():
                if name in loaded_params:
                    param.requires_grad = False

    def build_datasets(self):
        self.train_dataset = get_dataset(self.config, "train")
        self.val_dataset = get_dataset(self.config, "val")
        self.test_dataset = get_dataset(self.config, "test")

    def build_dataloaders(self):
        num_workers = self.config.setup.num_workers
        if num_workers == "auto":
            if n_cpu := os.environ.get("SLURM_CPUS_ON_NODE"):
                num_workers = int(n_cpu) // 2
            else:
                num_workers = os.cpu_count() // 2

        self.train_dataloader = DataLoader(
            self.train_dataset,
            batch_size = self.config.training.batch_size,
            shuffle = True,
            num_workers = num_workers,
            pin_memory = True,
        )
        self.val_dataloader = DataLoader(
            self.val_dataset,
            batch_size = self.config.training.batch_size,
            shuffle = False,
            num_workers = num_workers,
            pin_memory = True,
        )
        self.test_dataloader = DataLoader(
            self.test_dataset,
            batch_size = self.config.training.batch_size,
            shuffle = False,
            num_workers = num_workers,
            pin_memory = True,
        )

    def prepare_batch(self, batch):
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                v = v.to(self.device)
                if v.dtype.is_floating_point:
                    v = v.to(self.dtype)
                batch[k] = v
        return batch

    def log_end(self):
        self.logger.log_end()

    def log_step(self, loss):
        self.step += self.config.training.batch_size
        self.logger.log_scores({"train/loss": loss})

    def log_epoch(self, scores={}, **kwscores):
        match self.scheduler.get_last_lr():
            case [lr]:
                lrs = {"train/lr": lr}
            case [lr, finetune_lr]:
                lrs = {"train/lr": lr, "train/finetune_lr": finetune_lr}
            case _:
                lrs = {}

        scores = scores | kwscores | lrs
        self.logger.log_scores(scores)
        self.logger.save_state("latest")

        metric = "val/" + self.config.training.eval_metric
        metric_dir = self.config.training.eval_metric_direction
        if ((metric_dir == "min") and (scores[metric] < self.best_score)) or (
            (metric_dir == "max") and (scores[metric] > self.best_score)
        ):
            self.best_score = scores[metric]
            if self.config.training.get("save_best", True):
                self.logger.save_state("best")

        if self.epoch < self.config.training.epochs:
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
        self.use_gpu = self.device.type == "cuda"
        self.mixed = (self.config.setup.dtype == "mixed")
        match self.config.setup.dtype, self.use_gpu:
            case "bfloat16" | "bf16", True:
                self.dtype = torch.bfloat16
            case "float16" | "half" | "fp16" | "16" | 16, _:
                self.dtype = torch.float16
            case "float32" | "float" | "fp32" | "32" | 32, _:
                self.dtype = torch.float32
            case "mixed", _:
                self.dtype = torch.float32
            case _:
                raise ValueError(f"Invalid dtype selection: {self.config.setup.dtype}")

        if self.config.model == "fedformer":
            assert self.dtype == torch.float32, "Fedformer only supports float32 dtype"

        return self.dtype

    def handle_termination(self, signum, frame):
        print("Interrupted!")
        self.logger.save_state("latest")
        self.log_end()
        exit(0)

    @classmethod
    def from_run_id(cls, run_id, cfg=None, ckpt="latest", basepath=None):
        ckpt = ckpt or "latest"
        if basepath is None:
            basepath = Path(__file__).parent / f"../outputs/logs/{run_id}/"
        else:
            basepath = Path(basepath) / run_id

        config = toml.load(basepath / "config.toml")
        if cfg is not None:
            config = config | cfg
        config = dict_to_object(config)

        trainer = cls(run_id, config, newrun=False)

        modelpath = basepath / f"checkpoints/{ckpt}.pt"
        state = torch.load(modelpath)
        _, unexpected = trainer.model.load_state_dict(state["model"], strict=False)
        assert not unexpected, f"Unexpected keys in model state: {unexpected}"

        trainer.epoch = state["epoch"]
        trainer.step = state["step"]

        return trainer


class FineTuningScheduler(optim.lr_scheduler._LRScheduler):

    def __init__(self, optimizer, initial_lr, frozen_epochs, last_epoch=-1):
        self.initial_lr = initial_lr
        self.frozen_epochs = frozen_epochs
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        group_lrs = [group["lr"] for group in self.optimizer.param_groups]
        if self.last_epoch < self.frozen_epochs:
            return [group_lrs[0], 0]
        elif self.last_epoch == self.frozen_epochs:
            return [group_lrs[0], self.initial_lr]
        else:
            return group_lrs
