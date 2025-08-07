from typing import Literal
import torch
from torch import nn
from .mlp import MLPBase, mlp_forward


class PointNet(MLPBase):
    def __init__(
        self,
        in_dim: int, hid_dim: int, out_dim: int, depth: int,
        activation=nn.Tanh,
    ):
        '''
        ***CAUTION: `hid_dim` and `depth` will be ignored and replaced with the default PointNet configuration.***
        Qi et al. "PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation"
        '''
        nn.Module.__init__(self)
        MLPBase.__init__(self, in_dim, [64, 64, 64, 128, 1024, 512, 256, 128], out_dim, 8, activation=activation)

        self.encoder = nn.ModuleList([
            self._make_block(in_dim, 64, 64, activation=activation),
            self._make_block(64, 64, 128, 1024, activation=activation),
        ])
        self.decoder = self._make_block(1088, 512, 256, 128, activation=activation)
        self.head = nn.Linear(128, out_dim)
        
    @property
    def input_shape(self) -> Literal['flat', 'spatio-temporal']:
        return 'spatio-temporal'
        
    def _make_block(
        self,
        in_channels: int, 
        *hid_channels: tuple[int, ...], 
        activation=nn.ReLU, 
        use_bn: bool=False,
    ) -> nn.Sequential:
        layers = []
        channels = [in_channels] + list(hid_channels)
        for i in range(1, len(channels)):
            layers.append(nn.Conv1d(channels[i-1], channels[i], kernel_size=1))
            layers.append(activation())
        if use_bn:
            layers.append(nn.BatchNorm1d(channels[-1]))
        return nn.Sequential(*layers)

    @mlp_forward
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)  # ....................| T, in_dim, P
        enc_0 = self.encoder[0].forward(x)  # .......| T, 64, P
        enc_1 = self.encoder[1].forward(enc_0)  # ...| T, 1024, P
        
        pool = enc_1.max(dim=-1, keepdim=True).values
        pool = pool.expand_as(enc_1)  # .............| T, 1024, P
        
        enc = torch.cat([enc_0, pool], dim=1)  # ....| T, 1088, P
        dec = self.decoder.forward(enc)  # ..........| T, 128, P
        dec = dec.transpose(1, 2)  # ................| T, P, 128
        
        out = self.head.forward(dec)  # .............| T, P, 3
        return out.flatten(0, 1)  # .................| TP, 3
    

class PointNetLG(PointNet):    
    def __init__(
        self,
        in_dim: int, hid_dim: int, out_dim: int, depth: int,
        activation=nn.Tanh,
    ):
        '''
        ***CAUTION: `hid_dim` and `depth` will be ignored and replaced with the default PointNet configuration.***
        Qi et al. "PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation"
        '''
        PointNet.__init__(
            self, 
            in_dim=in_dim,
            hid_dim=hid_dim,
            out_dim=out_dim,
            depth=depth,
            activation=activation,
        )

        self.decoder_local = self._make_block(1024, 512, 256, 128, activation=nn.ReLU)
        self.head_local = nn.Linear(128, out_dim)
    
    @mlp_forward
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)  # ....................| T, in_dim, P
        enc_0 = self.encoder[0].forward(x)  # .......| T, 64, P
        enc_1 = self.encoder[1].forward(enc_0)  # ...| T, 1024, P
        
        # local branch forward
        dec_l = self.decoder_local.forward(enc_1)  # ...| T, 128, P
        dec_l = dec_l.transpose(1, 2)  # ...............| T, P, 128
        out_l = self.head_local.forward(dec_l)  # ......| T, P, 3
        out_l = out_l.flatten(0, 1)  # .................| TP, 3
        
        # global branch forward
        pool = enc_1.max(dim=-1, keepdim=True).values
        pool = pool.expand_as(enc_1)  # ..............| T, 1024, P
        
        enc_g = torch.cat([enc_0, pool], dim=1)  # ...| T, 1088, P
        dec_g = self.decoder.forward(enc_g)  # .......| T, 128, P
        dec_g = dec_g.transpose(1, 2)  # .............| T, P, 128
        
        out_g = self.head.forward(dec_g)  # ..........| T, P, 3
        out_g = out_g.flatten(0, 1)  # ...............| TP, 3

        return out_l, out_g

if __name__ == '__main__':
    model = PointNet(4, 64, 3, 8, nn.ReLU)
    sample_x = torch.randn(14, 128, 4)
    out = model.forward(sample_x)
    
    assert out.shape == (14, 128, 3)
    assert out.requires_grad
    print(PointNet, 'passed')
