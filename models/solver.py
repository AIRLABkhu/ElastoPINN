from typing import Type, Any, Generator, Literal
from abc import ABC, abstractmethod
import torch
from torch import nn
from .mlp import MLPOutput, MLPBase, MLP

T_RETURN_MODE = Literal[
    'mlp_output', 'tuple', 'global', 'local',
]


class Solver(nn.Module, ABC):
    def __init__(
        self,
        # specifying the model architecture
        in_dim: int = 4,
        hid_dim: int = 200, 
        out_dim: int = 3,
        depth: int = 6,
        activation: Type[nn.Module] = nn.Tanh,
        model_type: Type[MLPBase] = MLP,
        # configuring the environment
        ground_pos: float = 0.0,    # the ground will be at `y = ground_pos`
        gravity: float = 9.8,       # positive sign for downward
        up_index: int = 1,          # 0:x, 1:y, 2:z -> suppose `y` is the vertical axis`
    ):
        super().__init__()
        self.model = model_type(
            in_dim=in_dim, 
            hid_dim=hid_dim,
            out_dim=out_dim,
            depth=depth,
            activation=activation,
        )
        self._ground_pos = ground_pos
        self._gravity = gravity
        self._up_index = up_index
        
        self._config: dict[str, Any] = dict(
            in_dim=in_dim,
            hid_dim=hid_dim, 
            out_dim=out_dim,
            depth=depth,
            activation=activation,
            model_type=model_type,
            ground_pos=ground_pos, 
            gravity=gravity,
            up_index=up_index,
        )
        
        self._ret_mode: T_RETURN_MODE = 'mlp_output'
    
    def return_mode_(self, mode: T_RETURN_MODE) -> T_RETURN_MODE:
        self._ret_mode = mode
        return self._ret_mode
    
    @property
    def return_mode(self) -> T_RETURN_MODE:
        return self._ret_mode
        
    @property
    def ground_pos(self) -> float:
        return self._ground_pos
        
    @property
    def gravity(self) -> float:
        return self._gravity
        
    @property
    def up_index(self) -> int:
        return self._up_index
    
    @property
    def config(self) -> dict[str, Any]:
        return self._config
    
    def register_physical_property(self, name: str, value: float, optimized: bool):
        self._config[name] = value
        self._config[f'optim_{name}'] = optimized

    @abstractmethod
    def property_parameters(self) -> Generator[nn.Parameter, None, None]:
        ''' Returns the material properties parameters only. '''
        pass
    
    def network_parameters(self) -> Generator[nn.Parameter, None, None]:
        yield from self.model.parameters()

    def forward(self, x: torch.Tensor) -> MLPOutput:
        ret = self.model.forward(x)
        if self._ret_mode == 'mlp_output':
            return ret
        elif self._ret_mode == 'tuple':
            return ret.local_branch, ret.global_branch
        elif self._ret_mode == 'local':
            return ret.local_branch
        elif self._ret_mode == 'global':
            return ret.global_branch
        else:
            raise NotImplementedError

    @abstractmethod
    def compute_loss(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        pass
    
    def mode_scope(self, mode: T_RETURN_MODE):
        return SolverReturnMode(
            solver=self, mode=mode,
        )

class SolverReturnMode:
    def __init__(self, solver: Solver, mode: T_RETURN_MODE):
        self.solver = solver
        self.mode_original = self.solver.return_mode
        self.mode_scoped = mode
    
    def __enter__(self):
        self.solver.return_mode_(self.mode_scoped)
        return self
    
    def __exit__(self, *args, **kwargs):
        self.solver.return_mode_(self.mode_original)
