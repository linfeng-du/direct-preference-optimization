import inspect
import logging

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
    # Get the caller module name
    caller_frame = inspect.currentframe().f_back
    module_name = caller_frame.f_globals['__name__']

    logger = logging.getLogger(module_name)
    getattr(logger, level)(message)
