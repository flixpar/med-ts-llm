import toml

from tasks import get_trainer
from utils import *


def main():
    config = toml.load("config.toml")
    config = dict_to_object(config)

    run_id = get_run_id(config.DEBUG)
    trainer = get_trainer(run_id, config)

    trainer.train()

if __name__ == "__main__":
    main()
