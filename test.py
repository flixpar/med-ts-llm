import sys
from pathlib import Path
import toml

from tasks import task_lookup


def main(run_id, split="test", save_id=None, _basepath=None):
    basepath = Path(_basepath) if _basepath is not None else (Path(__file__).parent / "outputs/logs/")
    config = toml.load(basepath / run_id / "config.toml")
    task = config["task"]

    task_cls = task_lookup[task]
    trainer = task_cls.from_run_id(run_id, ckpt=save_id, basepath=_basepath)

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
        case [_, run_id, split, save_id, basepath]:
            main(run_id, split, save_id, basepath)
        case _:
            raise ValueError("Invalid number of arguments")
