from .forecasting import ForecastTask
from .anomaly_detection import AnomalyDetectionTask
from .reconstruction import ReconstructionTask
from .segmentation import SegmentationTask
from .semantic_segmentation import SemanticSegmentationTask
from .pretraining import PretrainingTask


task_lookup = {
    "forecasting": ForecastTask,
    "anomaly_detection": AnomalyDetectionTask,
    "reconstruction": ReconstructionTask,
    "segmentation": SegmentationTask,
    "semantic_segmentation": SemanticSegmentationTask,
    "pretraining": PretrainingTask,
}

def get_trainer(run_id, config):
    task_cls = task_lookup[config.task]
    return task_cls(run_id, config)
