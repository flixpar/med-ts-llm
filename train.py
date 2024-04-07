import sys
import toml

from tasks import get_trainer
from utils import *


def main(config_path, run_id=None):
    config = toml.load(config_path)
    config = dict_to_object(config)

    run_id = run_id or get_run_id(config.DEBUG)
    trainer = get_trainer(run_id, config)

    trainer.train()
    test_scores = trainer.test()
    trainer.log_end()

    print("Test results:", test_scores)
    print("Run ID:", run_id)


if __name__ == "__main__":
    match sys.argv:
        case [_, config_path, run_id]:
            main(config_path, run_id)
        case [_, config_path]:
            main(config_path)
        case _:
            main("configs/config.toml")
