import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.pytorch_utils import Conv1D


class LoRAController:
    adaptor_layers: list['LoRALayer']

    def __init__(self, r: int, alpha: int) -> None:
        self.r = r
        self.alpha = alpha
        self.adaptor_layers = []

    def insert_adaptors(self, model: nn.Module, target_modules: list[str]) -> None:
        """Replace target modules with adaptor layers and freeze the base model parameters."""
        for module_path, module in model.named_modules():
            module_name = module_path.split('.')[-1]

            if module_name in target_modules:
                assert isinstance(module, (nn.Linear, Conv1D)), 'Target module must be Linear or Conv1D'

                adaptor_layer = LoRALayer(module, self.r, self.alpha)
                self.adaptor_layers.append(adaptor_layer)

                # Replace the target module with an adaptor layer
                parent_module_path = '.'.join(module_path.split('.')[:-1])
                parent_module = model.get_submodule(parent_module_path)
                setattr(parent_module, module_name, adaptor_layer)

        # Freeze the base model parameters
        for param_path, param in model.named_parameters():
            if 'lora_A' not in param_path and 'lora_B' not in param_path:
                param.requires_grad = False


class LoRALayer(nn.Linear):

    def __init__(self, module: nn.Linear | Conv1D, r: int, alpha: int) -> None:
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
        self.scaling = alpha / r

        self.weight = weight
        self.bias = bias
        self.lora_A = nn.Parameter(self.weight.new_zeros(r, in_features))
        self.lora_B = nn.Parameter(self.weight.new_zeros(out_features, r))

        self.reset_parameters()

    def reset_parameters(self) -> None:
        if hasattr(self, 'lora_A'):
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        delta_w = self.lora_B @ self.lora_A * self.scaling
        w = self.weight + delta_w

        return F.linear(x, w, bias=self.bias)
