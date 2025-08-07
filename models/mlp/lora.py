from typing import Any, Optional
from functools import partial
import math
import torch
from torch import nn
from .mlp import MLPOutput, MLPBase, mlp_forward


class LoRALinear(nn.Linear):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        r: int = 4,
        bias: bool = True,
        device: Optional[Any] = None,
        dtype: Optional[Any] = None,
    ):
        super(LoRALinear, self).__init__(
            in_features=in_features,
            out_features=out_features,
            bias=bias,
            device=device,
            dtype=dtype,
        )
        self.r = r
        self.lora_A = nn.Parameter(torch.randn(in_features, r))
        self.lora_B = nn.Parameter(torch.randn(r, out_features))
        self.__mode = 'linear'  # 'linear' or 'lora'
        self.initialize_lora()
        
    @property
    def is_lora_mode(self) -> bool:
        return self.__mode == 'lora'
        
    def initialize_lora(self):
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)
    
    def linear(self):
        self.__mode = 'linear'
        self.weight.requires_grad_(True)
        if self.bias is not None:
            self.bias.requires_grad_(True)
        self.lora_A.requires_grad_(False)
        self.lora_B.requires_grad_(False)
        return self
    
    def lora(self):
        self.__mode = 'lora'
        self.weight.requires_grad_(False)
        if self.bias is not None:
            self.bias.requires_grad_(False)
        self.lora_A.requires_grad_(True)
        self.lora_B.requires_grad_(True)
        return self
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.is_lora_mode or (not self.training):
            lora_weight = self.weight + torch.mm(self.lora_A, self.lora_B).transpose(-2, -1)
            return nn.functional.linear(x, lora_weight, self.bias)
        else:
            return nn.functional.linear(x, self.weight, self.bias)

    def to_linear(self) -> nn.Linear:
        weight = self.weight + torch.mm(self.lora_A, self.lora_B).transpose(-2, -1)
        bias = self.bias
        linear = nn.Linear(
            self.in_features,
            self.out_features,
            bias=bias is not None,
            device=weight.device,
            dtype=weight.dtype,
        )
        linear.weight.data.copy_(weight)
        if bias is not None:
            linear.bias.data.copy_(bias)
        return linear


class LoRAMLP(MLPBase):
    def __init__(
        self,
        in_dim: int, hid_dim: int, out_dim: int, depth: int,
        activation=nn.Tanh,
    ):
        nn.Module.__init__(self)
        MLPBase.__init__(self, in_dim, hid_dim, out_dim, depth, activation)
        
        self.__mode = 'linear'
        self.layers = nn.Sequential(
            *self._make_stem_block(),
            *self._make_hidden_blocks(self.depth, layer_type=LoRALinear),
            self._make_linear_head(),
        )
        self.initialize_weights()
        self.initialize_siren()
        
    @property
    def is_lora_mode(self) -> bool:
        return self.__mode == 'lora'
    
    def linear(self):
        if self.is_lora_mode:
            self.__mode = 'linear'
            for module in self.modules():
                if isinstance(module, LoRALinear):
                    module.linear()
                elif isinstance(module, nn.Linear):
                    module.weight.requires_grad_(True)
                    if module.bias is not None:
                        module.bias.requires_grad_(True)
        return self
    
    def lora(self):
        if not self.is_lora_mode:
            self.__mode = 'lora'
            for module in self.modules():
                if isinstance(module, LoRALinear):
                    module.lora()
                elif isinstance(module, nn.Linear):
                    module.weight.requires_grad_(False)
                    if module.bias is not None:
                        module.bias.requires_grad_(False)
        return self
    
    @mlp_forward
    def forward(self, x: torch.Tensor) -> MLPOutput:
        return self.layers.forward(x)


if __name__ == '__main__':
    x = torch.randn(2, 4)
    model = LoRAMLP(4, 10, 3, 8)
    
    model.linear()
    y = model.forward(x).global_branch
    print(y.shape)
    
    model.lora()
    y = model.forward(x).global_branch
    print(y.shape)
    