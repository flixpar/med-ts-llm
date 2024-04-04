import sys
import toml

from tasks import task_lookup


def main(run_id, split="test", save_id=None):
    config = toml.load(f"outputs/logs/{run_id}/config.toml")
    task = config["task"]

    task_cls = task_lookup[task]
    trainer = task_cls.from_run_id(run_id, ckpt=save_id)

    if split == "test":
        test_scores = trainer.test()
    elif split == "val":
        test_scores = trainer.val()
    else:
        raise ValueError(f"Invalid split selected for testing: {split}")

    print("Results:", test_scores)
    print("Run ID:", run_id)

if __name__ == "__main__":
    match sys.argv:
        case [_, run_id]:
            main(run_id)
        case [_, run_id, split]:
            main(run_id, split)
        case [_, run_id, split, save_id]:
            main(run_id, split, save_id)
        case _:
            raise ValueError("Invalid number of arguments")
