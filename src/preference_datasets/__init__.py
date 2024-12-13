from typing import Any

from .persona import load_persona


def load_preference_dataset(dataset: str, split: str, **loader_kwargs: Any) -> (
    dict[str, dict[str, list[str] | list[float] | list[tuple[int, int]]]]
):
    dataset_configs = {
        'persona': {'loader': load_persona, 'truncation_mode': 'keep_end'}
    }
    data = dataset_configs[dataset]['loader'](split, **loader_kwargs)
    truncation_mode = dataset_configs[dataset]['truncation_mode']
    return data, truncation_mode
