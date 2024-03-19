import sys
import toml

from tasks import get_trainer
from utils import *


def main(config_path):
    config = toml.load(config_path)
    config = dict_to_object(config)

    run_id = get_run_id(config.DEBUG)
    trainer = get_trainer(run_id, config)

    trainer.train()
    test_scores = trainer.test()
    trainer.log_end()

    print("Test results:", test_scores)
    print("Run ID:", run_id)

if __name__ == "__main__":
    config_path = sys.argv[1] if len(sys.argv) > 1 else "configs/config.toml"
    main(config_path)
