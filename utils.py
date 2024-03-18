import torch
import random
from datetime import datetime


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
