from torch.utils.tensorboard import SummaryWriter

from .base_logger import BaseLogger
from utils import summarize_config, flatten_dict


class TensorboardLogger(BaseLogger):
    def __init__(self, trainer, config, newrun):
        super().__init__(trainer, config, newrun)

        self.logger = SummaryWriter(log_dir=self.logdir / "tensorboard")

        cfg = flatten_dict(summarize_config(self.config))
        cfg = {k: (v if not isinstance(v, list) else ", ".join(v)) for k, v in cfg.items()}
        self.logger.add_hparams(cfg, {}, run_name=".")

    def log_end(self):
        self.logger.close()

    def log_scores(self, scores={}, **kwscores):
        self.logger.add_scalar("epoch", self.trainer.epoch, self.trainer.step)

        scores = scores | kwscores
        for key, value in scores.items():
            self.logger.add_scalar(key, value, self.trainer.step)

    def update_config(self, cfg):
        super().update_config(cfg)
        self.logger.add_hparams(flatten_dict(cfg), {}, run_name=".")
