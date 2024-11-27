import math

import torch
import torch.nn as nn
from transformers.pytorch_utils import Conv1D

from typing import Union


class MixtureOfLoRAsController:
    model: nn.Module
    adaptor_layers: list['MixtureOfLoRAsLayer']

    def __init__(self, model: nn.Module) -> None:
        self.model = model
        self.adaptor_layers = []

    def insert_adaptors(self, target_modules: list[str], n_loras: int, r: int, alpha: int) -> None:
        for module_path, module in self.model.named_modules():
            module_name = module_path.split('.')[-1]

            if module_name in target_modules:
                assert isinstance(module, (nn.Linear, Conv1D))
                adaptor_layer = MixtureOfLoRAsLayer(module, n_loras, r, alpha)
                self.adaptor_layers.append(adaptor_layer)

                parent_module_path = '.'.join(module_path.split('.')[:-1])
                parent_module = self.model.get_submodule(parent_module_path)
                setattr(parent_module, module_name, adaptor_layer)

        for param_name, param in self.model.named_parameters():
            if param.requires_grad and 'lora' not in param_name:
                param.requires_grad = False

    def update_lora_weights(self, lora_weights: torch.Tensor) -> None:
        for adaptor_layer in self.adaptor_layers:
            adaptor_layer.update_lora_weights(lora_weights)


class MixtureOfLoRAsLayer(nn.Linear):

    def __init__(self, module: Union[nn.Linear, Conv1D], n_loras: int, r: int, alpha: int) -> None:
        if isinstance(module, nn.Linear):
            in_features = module.in_features
            out_features = module.out_features
            weight = module.weight
            bias = module.bias
        elif isinstance(module, Conv1D):
            in_features = module.nx
            out_features = module.nf
            weight = nn.Parameter(module.weight.T)
            bias = module.bias

        super().__init__(in_features, out_features)
        self.n_loras = n_loras
        self.scaling = alpha / r

        self.weight = weight
        self.bias = bias
        self.lora_A = nn.Parameter(self.weight.new_zeros(self.n_loras, r, in_features))
        self.lora_B = nn.Parameter(self.weight.new_zeros(self.n_loras, out_features, r))

        self.lora_weights = None
        self.reset_parameters()

    def reset_parameters(self) -> None:
        if hasattr(self, 'lora_A'):
            for i in range(self.n_loras):
                nn.init.kaiming_uniform_(self.lora_A[i], a=math.sqrt(5))

    def update_lora_weights(self, lora_weights: torch.Tensor) -> None:
        self.lora_weights = lora_weights

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w0 = self.weight.unsqueeze(0).expand(x.size(0), -1, -1)
        delta_w = torch.einsum(
            'bn,noi->boi',
            self.lora_weights,
            self.lora_B @ self.lora_A
        ) * self.scaling
        w = (w0 + delta_w).mT

        out = torch.bmm(x, w)
        if self.bias is not None:
            out += self.bias

        return out
