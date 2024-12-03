import random
import inspect
import logging
from types import TracebackType

import numpy as np

import torch
import torch.nn as nn
from accelerate import PartialState


@PartialState().on_main_process
def log_main_process(message: str, level: str = 'info') -> None:
    """Log only on main process."""
    logger = _get_logger_with_context()
    getattr(logger, level)(message)


def log_all_processes(message: str, level: str = 'info') -> None:
    """Log on all processes."""
    logger = _get_logger_with_context()
    getattr(logger, level)(message)


def _get_logger_with_context() -> logging.Logger:
    """Get the logger for the calling module."""
    caller_frame = inspect.currentframe().f_back
    module_name = caller_frame.f_globals['__name__']
    return logging.getLogger(module_name)


def formatted_dict(d: dict) -> dict:
    """Format a dictionary for printing."""
    return {k: (f"{v:.5g}" if type(v) == float else v) for k, v in d.items()}


def disable_dropout(model: nn.Module) -> None:
    """Disable dropout in a model."""
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.p = 0.


def slice_and_move_batch_for_device(batch: dict, rank: int, world_size: int, device: str) -> dict:
    """Slice a batch into chunks, and move each chunk to the specified device."""
    if 'user_emb' in batch:
        batch['user_emb'] = torch.tensor(batch['user_emb']).to(device)
    chunk_size = len(list(batch.values())[0]) // world_size
    start = chunk_size * rank
    end = chunk_size * (rank + 1)
    sliced = {k: v[start:end] for k, v in batch.items()}
    on_device = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in sliced.items()}
    return on_device


def pad_to_length(tensor: torch.Tensor, length: int, pad_value: int | float, dim: int = -1) -> torch.Tensor:
    if tensor.size(dim) >= length:
        return tensor
    else:
        pad_size = list(tensor.shape)
        pad_size[dim] = length - tensor.size(dim)
        return torch.cat([tensor, pad_value * torch.ones(*pad_size, dtype=tensor.dtype, device=tensor.device)], dim=dim)


class TemporarilySeededRandom:
    """Context manager for controlled randomness.

    Sets the random seed and restores the original seed when exiting.
    """

    def __init__(self, seed: int) -> None:
        self.seed = seed
        self.stored_state = None
        self.stored_np_state = None

    def __enter__(self) -> None:
        # Store the current random states
        self.stored_state = random.getstate()
        self.stored_np_state = np.random.get_state()

        # Set the random seed
        random.seed(self.seed)
        np.random.seed(self.seed)

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None
    ) -> None:
        # Restore the original random states
        random.setstate(self.stored_state)
        np.random.set_state(self.stored_np_state)
