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
    """Log only on the main process."""
    # Get the caller module name
    caller_frame = inspect.currentframe().f_back
    module_name = caller_frame.f_globals['__name__']

    logger = logging.getLogger(module_name)
    getattr(logger, level)(message)


def log_all_processes(message: str, level: str = 'info') -> None:
    """Log on all processes."""
    # Get caller module name
    caller_frame = inspect.currentframe().f_back
    module_name = caller_frame.f_globals['__name__']

    logger = logging.getLogger(module_name)
    getattr(logger, level)(message)


def formatted_dict(d: dict) -> dict:
    """Format a dictionary for printing."""
    return {k: (f"{v:.5g}" if type(v) == float else v) for k, v in d.items()}


def disable_dropout(model: nn.Module) -> None:
    """Disable dropout in a model."""
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.p = 0.


class TemporarilySeededRandom:
    """Context manager for controlled randomness.
    Set the random seed and restore the original random states when exiting.
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
