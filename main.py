import toml

from tasks import ForecastTask
from utils import *


def main():
    config = toml.load("config.toml")
    config = dict_to_object(config)

    run_id = get_run_id(config.DEBUG)
    trainer = ForecastTask(run_id=run_id, config=config)

    trainer.train()

if __name__ == "__main__":
    main()
