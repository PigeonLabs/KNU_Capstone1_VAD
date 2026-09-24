"""Run provenance and deterministic experiment utilities."""
import json
import os
import random
from pathlib import Path

import numpy as np
import torch


SCENES = ('R01', 'R02', 'R03', 'R04')
PAPER_AUC = dict(zip(SCENES, (84.4, 75.4, 43.5, 76.7)))


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    os.replace(temp, path)


def prepare_clip(batch, device):
    # Official model consumes OpenCV BGR in [-1, 1]. Dataset returns uint8.
    return batch.to(device, non_blocking=True).permute(0, 4, 1, 2, 3).float().div_(127.5).sub_(1)


def atomic_checkpoint(path, state):
    path = Path(path)
    temp = path.with_suffix('.tmp')
    torch.save(state, temp)
    os.replace(temp, path)


def environment():
    import platform
    return {'python': platform.python_version(), 'torch': torch.__version__,
            'cuda': torch.version.cuda, 'gpu': torch.cuda.get_device_name() if torch.cuda.is_available() else None,
            'capability': list(torch.cuda.get_device_capability()) if torch.cuda.is_available() else None}
