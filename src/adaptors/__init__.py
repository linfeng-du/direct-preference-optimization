from typing import Any

from .lora import LoRAController
from .mixture_of_loras import MixtureOfLoRAsController


Controller = LoRAController | MixtureOfLoRAsController


def get_controller(adaptor: str, **adaptor_kwargs: Any) -> Controller:
    controller_classes = {
        'lora': LoRAController,
        'mixture_of_loras': MixtureOfLoRAsController
    }
    return controller_classes[adaptor](**adaptor_kwargs)
