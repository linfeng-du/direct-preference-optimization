from .mixture_of_loras import MixtureOfLoRAsController


def get_controller_class(adaptor: str) -> MixtureOfLoRAsController:
    controller_classes = {
        'mixture_of_loras': MixtureOfLoRAsController
    }
    return controller_classes[adaptor]
