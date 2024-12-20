import random
import inspect
import logging

import numpy as np

import torch
import torch.nn as nn
from accelerate.state import PartialState


state = PartialState()


def log_accelerate(message: str, level: str = 'info', on_all_processes: bool = False) -> None:
    """Log on all processes."""
    def log(message, level):
        # Get the caller module name
        caller_frame = inspect.currentframe().f_back.f_back
        module_name = caller_frame.f_globals['__name__']

        logger = logging.getLogger(module_name)
        getattr(logger, level)(message)

    if on_all_processes:
        log(message, level)
    elif state.is_local_main_process:
        log(message, level)


def seed_everything(seed: int) -> None:
    """Seed everything for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def disable_dropout(model: nn.Module) -> None:
    """Disable dropout in the model."""
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.p = 0.


def count_trainable_parameters(model: nn.Module) -> str:
    """Count the number of trainable parameters of the model."""
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    if num_params >= 1e9:
        num_params = f'{num_params / 1e9:.2f}B'
    elif num_params >= 1e6:
        num_params = f'{num_params / 1e6:.2f}M'
    else:
        num_params = f'{num_params:,}'

    return num_params
