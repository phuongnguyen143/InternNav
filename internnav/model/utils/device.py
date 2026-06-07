import logging

import torch

logger = logging.getLogger(__name__)


def default_torch_device() -> torch.device:
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def resolve_torch_device(device=None) -> torch.device:
    if device is None:
        return default_torch_device()

    if isinstance(device, torch.device):
        resolved = device
    elif isinstance(device, int):
        resolved = torch.device(f'cuda:{device}')
    else:
        resolved = torch.device(device)

    if resolved.type == 'cuda' and not torch.cuda.is_available():
        logger.warning('CUDA is not available; using CPU instead (requested %s)', device)
        return torch.device('cpu')
    return resolved


def model_load_dtype(device: torch.device) -> torch.dtype:
    return torch.bfloat16 if device.type == 'cuda' else torch.float32
