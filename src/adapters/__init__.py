from typing import Any

from .lora import LoRAController
from .mixture_of_loras import MixtureOfLoRAsController


def get_controller(adapter: str, **adapter_kwargs: Any) -> (
    LoRAController | MixtureOfLoRAsController
):
    controller_classes = {
        'lora': LoRAController,
        'mixture_of_loras': MixtureOfLoRAsController
    }
    return controller_classes[adapter](**adapter_kwargs)
