from .forecasting import ForecastTask


def get_trainer(run_id, config):
    match config.task:
        case "forecasting":
            return ForecastTask(run_id=run_id, config=config)
        case _:
            raise ValueError(f"Invalid task selection: {config.task}")
