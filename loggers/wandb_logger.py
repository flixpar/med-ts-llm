import wandb
from pathlib import Path

from .base_logger import BaseLogger
from utils import get_logging_tags, summarize_config


class WandBLogger(BaseLogger):
    def __init__(self, trainer, config, newrun=True):
        super().__init__(trainer, config, newrun)

        self.logger = wandb.init(
            project = "med-time-llm",
            entity = "med-time-llm",
            name = self.trainer.run_id,
            id = self.trainer.run_id,
            dir = self.logdir,
            config = summarize_config(self.config),
            tags = get_logging_tags(self.config),
            resume = "allow",
            job_type = "training",
            mode = "online" if not self.config.DEBUG else "disabled",
        )
        self.log_code()

    def log_end(self):
        self.logger.finish()

    def log_scores(self, scores={}, **kwscores):
        self.logger.log({"epoch": self.trainer.epoch, "step": self.trainer.step} | scores | kwscores)

    def update_config(self, cfg):
        super().update_config(cfg)
        self.logger.config.update(cfg)

    def log_code(self):
        basepath = Path(__file__).parent.parent
        excluded_dirs = [
            basepath / ".wandb",
            basepath / "wandb",
            basepath / ".venv",
            basepath / "tmp",
            basepath / "backup",
        ]

        def exclude_fn(path, root):
            path = Path(root) / path
            for excluded_dir in excluded_dirs:
                if excluded_dir in path.parents:
                    return True
            return False

        self.logger.log_code(exclude_fn=exclude_fn)
