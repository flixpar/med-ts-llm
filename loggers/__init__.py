from .wandb_logger import WandBLogger
from .tensorboard_logger import TensorboardLogger
from .print_logger import PrintLogger
from .debug_logger import DebugLogger


def get_logger(trainer, config, newrun=True):
    if config.DEBUG:
        return DebugLogger(trainer, config, newrun)
    match config.setup.logger:
        case "wandb":
            return WandBLogger(trainer, config, newrun)
        case "tensorboard":
            return TensorboardLogger(trainer, config, newrun)
        case "print" | "none":
            return PrintLogger(trainer, config, newrun)
        case _:
            raise ValueError(f"Unknown logger: {config.setup.logger}")
