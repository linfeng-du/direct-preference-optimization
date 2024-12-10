import random
from types import TracebackType

import numpy as np


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
