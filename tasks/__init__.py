from .forecasting import ForecastTask
from .anomaly_detection import AnomalyDetectionTask
from .segmentation import SegmentationTask
from .semantic_segmentation import SemanticSegmentationTask


task_lookup = {
    "forecasting": ForecastTask,
    "anomaly_detection": AnomalyDetectionTask,
    "segmentation": SegmentationTask,
    "semantic_segmentation": SemanticSegmentationTask,
}

def get_trainer(run_id, config):
    task_cls = task_lookup[config.task]
    return task_cls(run_id, config)
