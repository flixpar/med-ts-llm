from .wandb_logger import WandBLogger
from .tensorboard_logger import TensorboardLogger


def get_logger(trainer, config, newrun=True):
    match config.setup.logger:
        case "wandb":
            return WandBLogger(trainer, config, newrun)
        case "tensorboard":
            return TensorboardLogger(trainer, config, newrun)
        case _:
            raise ValueError(f"Unknown logger: {config.setup.logger}")
