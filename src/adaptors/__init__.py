from .lora import LoRAController
from .mixture_of_loras import MixtureOfLoRAsController


def get_controller_class(adaptor: str) -> LoRAController | MixtureOfLoRAsController:
    controller_classes = {
        'lora': LoRAController,
        'mixture_of_loras': MixtureOfLoRAsController
    }
    return controller_classes[adaptor]
