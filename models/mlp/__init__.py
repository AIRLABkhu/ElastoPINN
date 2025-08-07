from typing import Type
from .mlp import MLPOutput, MLPBase, MLP, GlobalMLP
from .time_attn_mlp import TimeAttnMLP
from .transformer import AttnMLP
from .pointnet import PointNet, PointNetLG
from .lora import LoRAMLP


mlp_dict: dict[str, Type[MLPBase]] = {
    'mlp': MLP,
    'globalmlp': GlobalMLP,
    'timeattnmlp': TimeAttnMLP,
    'attnmlp': AttnMLP,
    'pointnet': PointNet,
    'pointnetlg': PointNetLG,
    'loramlp': LoRAMLP,
}

del Type
