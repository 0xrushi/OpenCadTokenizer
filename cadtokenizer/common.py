from __future__ import annotations

import hashlib
import json
import os
import random
from pathlib import Path


def model_runtime_available():
    if not Path('/.dockerenv').exists():
        return False
    return (os.environ.get('CADTOKENIZER_CONTAINER') == 'unsloth'
            or (os.environ.get('CADTOKENIZER_RUNTIME') == 'vastai'
                and bool(os.environ.get('VAST_CONTAINERLABEL'))))


def require_container():
    if not model_runtime_available():
        raise RuntimeError('Use ./scripts/cad.sh locally or scripts/vast.sh inside the authorized Vast.ai instance.')


def read_config(path):
    with open(path) as f:
        return json.load(f)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def file_hash(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as f:
        while chunk := f.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def fraction(key, seed=42):
    return int(hashlib.sha256(f'{seed}:{key}'.encode()).hexdigest()[:16], 16) / 2**64


def seed_all(seed):
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def device_for(config):
    import torch
    requested = config.get('device', 'cuda')
    if requested == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable. Check Docker GPU access with ./scripts/cad.sh doctor. Use the smoke config for CPU checks.')
    return torch.device(requested)


def load_checkpoint(path):
    import torch
    return torch.load(path, map_location='cpu', weights_only=True)


def save_checkpoint(path, state):
    import torch
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    torch.save(state, tmp)
    tmp.replace(path)
