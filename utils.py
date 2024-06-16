import torch
import random
from datetime import datetime
from copy import deepcopy


def get_run_id(debug=False):
    run_id = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    if debug:
        run_id = "DEBUG-" + run_id
    return run_id


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)


class dict_to_object(object):
    def __init__(self, d):
        self.__dict__ = {k: dict_to_object(v) if isinstance(v, dict) else v for k, v in d.items()}

    def to_dict(self):
        return {k: v.to_dict() if isinstance(v, dict_to_object) else v for k, v in self.__dict__.items()}

    def get(self, key, default=None):
        return self.__dict__.get(key, default)

    def __getitem__(self, key):
        return self.__dict__[key]

    def __contains__(self, key):
        return key in self.__dict__
    
    def __repr__(self):
        return str(self.__dict__)

    def copy(self):
        return deepcopy(self)


def get_logging_tags(config):
    tags = [
        "data:" + config.data.dataset,
        "task:" + config.task,
        "model:" + config.model,
    ]
    return tags

def summarize_config(config):
    config = deepcopy(config.to_dict())

    model = config["model"]
    task = config["task"]
    dataset = config["data"]["dataset"]

    if model in config["models"]:
        model_config = config["models"][model]
        config[model] = model_config
        del config["models"]

    if "tasks" in config:
        for t in list(config["tasks"].keys()):
            if t != task:
                del config["tasks"][t]

    if "datasets" in config and config["data"]["dataset"] != "all":
        for d in list(config["datasets"].keys()):
            if d != dataset:
                del config["datasets"][d]

    return config


def flatten_dict(d, parent_key="", sep="."):
    output = {}
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            output = output | flatten_dict(v, new_key, sep)
        else:
            output[new_key] = v
    return output


def get_dtype(dtype_name):
    match dtype_name:
        case "bfloat16" | "bf16":
            return torch.bfloat16
        case "float16" | "half" | "fp16" | "16" | 16:
            return torch.float16
        case "float32" | "float" | "fp32" | "32" | 32 | "mixed":
            return torch.float32
        case x:
            raise ValueError(f"Invalid dtype selection: {x}")
