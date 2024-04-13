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

        if basepath := config.get("paths", {}).get("logdir"):
            basepath = Path(basepath)
        else:
            basepath = Path(__file__).parent / "../outputs/logs/"

        self.logdir = basepath / self.trainer.run_id
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

    def update_config(self, cfg):
        if not isinstance(cfg, dict):
            cfg = cfg.to_dict()

        if (self.logdir / "config-updates.toml").exists():
            with open(self.logdir / "config-updates.toml", "r") as f:
                cfg = toml.load(f) | cfg

        with open(self.logdir / "config-updates.toml", "w") as f:
            toml.dump(cfg, f)
        with open(self.logdir / "config-updates.json", "w") as f:
            json.dump(cfg, f, indent="\t")

    @abstractmethod
    def log_end(self):
        pass

    @abstractmethod
    def log_scores(self, scores={}, **kwscores):
        pass

    def log_figure(self, fig, name):
        pass
