from .wandb_logger import WandBLogger
from .tensorboard_logger import TensorboardLogger
from .print_logger import PrintLogger


def get_logger(trainer, config, newrun=True):
    if config.DEBUG:
        return PrintLogger(trainer, config, newrun)
    match config.setup.logger:
        case "wandb":
            return WandBLogger(trainer, config, newrun)
        case "tensorboard":
            return TensorboardLogger(trainer, config, newrun)
        case "print" | "none":
            return PrintLogger(trainer, config, newrun)
        case _:
            raise ValueError(f"Unknown logger: {config.setup.logger}")
