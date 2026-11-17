import random
from pathlib import Path
import numpy as np
import torch
import yaml


def set_seed(s):
    random.seed(s);np.random.seed(s);torch.manual_seed(s)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(s)


def load_config(path):
    with open(path) as f:return yaml.safe_load(f)


def device_from_arg(value):
    return torch.device(value if value!='auto' else ('cuda' if torch.cuda.is_available() else 'cpu'))


def to_device(batch,device):
    return {k:(v.to(device) if torch.is_tensor(v) else v) for k,v in batch.items()}


def makedirs(p):
    Path(p).mkdir(parents=True,exist_ok=True)
