from hnet.lora import LoRA,LoRA_controller,LoRAHnet
from hnet.hypernet import HyperNet,HyperNetController
from hnet.loraHnet import LoRAHnet_controller,LoRAHnet


def get_module(net,config):
    if net == 'lora':
        net = LoRAHnet(config)
        return net,LoRA_controller(config)
    elif net == 'hypernet':
        net = HyperNet(config)
        return net,HyperNetController(config,net)
    elif net == 'lora_hnet':
        net = LoRAHnet(10,10)
        return net,LoRAHnet_controller(config)