from abc import ABC, abstractmethod
from pathlib import Path
import toml, json
from datetime import datetime
import torch


class BaseLogger(ABC):
    def __init__(self, trainer, config, newrun=True):
        self.trainer = trainer
        self.config = config
        self.newrun = newrun

        self.logdir = Path(__file__).parent / f"../outputs/logs/{self.trainer.run_id}/"
        self.logdir.mkdir(parents=True, exist_ok=True)

        if self.newrun:
            config_dict = self.config.to_dict()
            with open(self.logdir / "config.toml", "w") as f:
                toml.dump(config_dict, f)
            with open(self.logdir / "config.json", "w") as f:
                json.dump(config_dict, f, indent="\t")

    def save_state(self, name):
        modeldir = self.logdir / "checkpoints"
        modeldir.mkdir(parents=True, exist_ok=True)

        state = {
            "run_id": self.trainer.run_id,
            "epoch": self.trainer.epoch,
            "step": self.trainer.step,
            "datetime": datetime.now().isoformat(),
            "model": self.trainer.model.state_dict(),
        }
        torch.save(state, modeldir / f"{name}.pt")

    @abstractmethod
    def log_end(self):
        pass

    @abstractmethod
    def log_scores(self, scores={}, **kwscores):
        pass
