import torch
from torch import nn
from pytorch3d.ops import sample_farthest_points, knn_points
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
        self,
        q: torch.Tensor,
        k: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        emb_q = self.proj_q.forward(q)  # ...| (NQ, rD)
        emb_k = self.proj_k.forward(k)  # ...| (NK, rD)
        emb_v = self.proj_v.forward(k)  # ...| (NK, D)
        
        attn = torch.mm(emb_q, emb_k.T) / self.scale  # ...| (NQ, NK)
        attn = torch.softmax(attn, dim=-1) 
        
        out = self.activation(torch.mm(attn, emb_v))  # ...| (NQ, D)
        return out, attn


class AttnMLP(MLPBase):
    def __init__(
        self,
        in_dim: int, hid_dim: int, out_dim: int, depth: int,
        fps_ratio: float=0.2,  # for local-attention
        activation=nn.Tanh,
    ):
        nn.Module.__init__(self)
        MLPBase.__init__(self, in_dim, hid_dim, out_dim, depth, activation)
        self.fps_ratio = fps_ratio
        
        depth_0 = self.depth // 2
        depth_1 = self.depth - depth_0 - 1
        
        self.layer_input = nn.Sequential(*self._make_stem_block())
        self.layer_hidden_0 = nn.Sequential(*self._make_hidden_blocks(depth_0))
        self.layer_attention = Attention(hid_dim, r=0.5, activation=activation)
        self.layer_hidden_1 = nn.Sequential(*self._make_hidden_blocks(depth_1))
        self.layer_output = self._make_linear_head()
        
        self.initialize_weights() 
        self.initialize_siren()
    
    @mlp_forward
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_i = self.layer_input.forward(x)
        x_h0 = self.layer_hidden_0.forward(x_i)
        
        # fps - attention
        k = int(len(x) * self.fps_ratio)
        fps_out = sample_farthest_points(x_h0[None], K=k)
        x_centers = fps_out[0][0]
        x_at, _ = self.layer_attention.forward(
            x_centers, x_h0,
        )  # (k, D)
        
        # broadcast residuals
        with torch.no_grad():
            knn_out = knn_points(x_h0[None], x_centers[None], K=1)
            indices = knn_out.idx[0].squeeze(-1)
        x_at = x_at[indices] + x_h0  # (B, D)
        
        x_h1 = self.layer_hidden_1.forward(x_at)
        x_o = self.layer_output.forward(x_h1)
        return x_o


if __name__ == '__main__':
    sample_x = torch.randn(10, 4)
    target_shape = (10, 3)

    for mlp_type in [AttnMLP]:
        model = mlp_type(4, 200, 3, 6)
        out = model.forward(sample_x)
        assert out.shape == target_shape
        assert out.requires_grad
        print(mlp_type, 'passed')
