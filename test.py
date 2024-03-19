import sys
import toml

from tasks import task_lookup


def main(run_id):
    config = toml.load(f"outputs/logs/{run_id}/config.toml")
    task = config["task"]

    task_cls = task_lookup[task]
    trainer = task_cls.from_run_id(run_id)

    test_scores = trainer.test()

    print("Test results:", test_scores)
    print("Run ID:", run_id)

if __name__ == "__main__":
    assert len(sys.argv) == 2
    run_id = sys.argv[1]
    main(run_id)
