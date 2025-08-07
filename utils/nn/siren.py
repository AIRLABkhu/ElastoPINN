import torch
from torch import nn


class SIREN(nn.Module):
    def __init__(self, omega_0: float=30.0, is_first: bool=False):
        super().__init__()
        self._omega_0 = omega_0
        self._is_first = is_first
        
        if is_first:
            self._omega = omega_0
        else:
            self._omega = 1.0
    
    @property
    def is_first(self) -> bool:
        return self._is_first
    
    @is_first.setter
    def is_first(self, value: bool):
        self._is_first = value
        self._omega = self._omega_0 if value else 1.0
        
    @property
    def omega(self) -> float:
        return self._omega
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sin(self.omega * x)
    
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x)
