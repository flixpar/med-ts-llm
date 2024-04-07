import json

from .base_logger import BaseLogger
from utils import summarize_config


class DebugLogger(BaseLogger):
    def __init__(self, trainer, config, newrun=True):
        self.trainer = trainer
        self.config = config

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

    def save_state(self, name):
        pass
