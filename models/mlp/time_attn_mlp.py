from typing import Literal
import torch
from torch import nn
from .mlp import MLPBase, mlp_forward


class Attention(nn.Module):
    def __init__(self, dim: int, r: float=1.0, activation=nn.Tanh):
        super().__init__()
        self.dim = dim
        self.activation_type = activation
        
        dim_r = int(dim * r)
        self.proj_q = nn.Linear(dim, dim_r)
        self.proj_k = nn.Linear(dim, dim_r)
        self.proj_v = nn.Linear(dim, dim)
        self.activation = self.activation_type()
        self.scale = dim ** 0.5
    
    def forward(
        self, x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        emb_q = self.proj_q.forward(x)  # ...| (T, P, rD)
        emb_k = self.proj_k.forward(x)  # ...| (T, P, rD)
        emb_v = self.proj_v.forward(x)  # ...| (T, P, D)

        emb_q = emb_q.permute(1, 0, 2)  # ...| (P, T, rD)
        emb_k = emb_k.permute(1, 0, 2)  # ...| (P, T, rD)
        emb_v = emb_v.permute(1, 0, 2)  # ...| (P, T, D)

        attn = torch.matmul(emb_q, emb_k.transpose(-2, -1)) / self.scale
        attn = torch.softmax(attn, dim=-1)  # ...| (P, T, T)

        attn_output = torch.matmul(attn, emb_v)  # ...| (P, T, D)
        return attn_output.permute(1, 0, 2)  # .......| (T, P, D)


class TimeAttnMLP(MLPBase):
    def __init__(
        self,
        in_dim: int, hid_dim: int, out_dim: int, depth: int,
        activation=nn.Tanh,
    ):
        nn.Module.__init__(self)
        MLPBase.__init__(self, in_dim, hid_dim, out_dim, depth, activation)
        
        depth_0 = self.depth // 2
        depth_1 = self.depth - depth_0 - 1
        
        self.layer_input = nn.Sequential(*self._make_stem_block())
        self.layer_hidden_0 = nn.Sequential(*self._make_hidden_blocks(depth_0))
        self.layer_attention = Attention(hid_dim, r=0.5, activation=activation)
        self.layer_hidden_1 = nn.Sequential(*self._make_hidden_blocks(depth_1))
        self.layer_output = self._make_linear_head()
        
        self.initialize_weights() 
        self.initialize_siren()
        
    @property
    def input_shape(self) -> Literal['flat', 'spatio-temporal']:
        return 'spatio-temporal'
    
    @mlp_forward
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_i = self.layer_input.forward(x)
        x_h0 = self.layer_hidden_0.forward(x_i)
        x_at = self.layer_attention.forward(x_h0) + x_h0
        x_h1 = self.layer_hidden_1.forward(x_at)
        x_o = self.layer_output.forward(x_h1)
        return x_o.flatten(0, 1)


if __name__ == '__main__':
    sample_x = torch.randn(5, 2, 4)
    target_shape = (10, 3)

    for mlp_type in [TimeAttnMLP]:
        model = mlp_type(4, 200, 3, 6)
        out = model.forward(sample_x)
        assert out.shape == target_shape
        assert out.requires_grad
        print(mlp_type, 'passed')
