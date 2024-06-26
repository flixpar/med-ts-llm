import json

from .base_logger import BaseLogger
from utils import summarize_config


class PrintLogger(BaseLogger):
    def __init__(self, trainer, config, newrun=True):
        super().__init__(trainer, config, newrun)

        config = summarize_config(config)
        config = json.dumps(config, indent="\t")

        print("Run ID:", trainer.run_id)
        print("Config:")
        print(config)

    def log_end(self):
        print("Done!")

    def log_scores(self, scores={}, **kwscores):
        scores = scores | kwscores
        if len(scores) == 1 and "train/loss" in scores:
            return
        print(f"Epoch: {self.trainer.epoch}, step: {self.trainer.step}, scores: {scores}")

    def update_config(self, cfg):
        super().update_config(cfg)
        print("Config updated:", cfg)
