from argparse import ArgumentParser
from tqdm import tqdm
import inspect

import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from configs.config import CFG as cfg
from data.pac_nerf import PACNeRFDataset

from models.mlp import mlp_dict
from models.navier_cauchy import NavierCauchy
from utils.logging import (
    Averager,
    CheckpointWriter,
)
from utils.nn import SIREN
from utils import gc

torch.autograd.set_detect_anomaly(True)

# -------------------------------------
# Load configuration
# -------------------------------------

MLP_INIT_DICT = {
    'xavier_uni': nn.init.xavier_uniform_,
    'xavier_nor': nn.init.xavier_normal_,
    'kaiming_uni': nn.init.kaiming_uniform_,
    'kaiming_nor': nn.init.kaiming_normal_,
}

parser = ArgumentParser('Navier-Cauchy')
add = parser.add_argument
add('--device', '-d', type=int, default=0)
add('--object', '-o', type=str.lower, default='bird', choices=PACNeRFDataset.INSTANCES)
add('--mlp', '-mlp', type=str.lower, default='mlp', choices=mlp_dict.keys())
add('--mlp-init', type=str.lower, default='xavier_nor', choices=list(MLP_INIT_DICT.keys()))
add('--num-frames', '-nf', type=int, default=14)
add('--batch-size', '-bs', type=int, default=20000)
add('--learning-rate', '-lr', type=float, default=1.0E-4)
add('--property-learning-rate', '-plr', type=float, default=1.0E-1)
add('--epochs', '-e', type=int, default=10_000)
add('--save-every', type=int, default=1_000)
add('--tag', type=str, default=None)
add('--overwrite', action='store_true', default=False)
args = parser.parse_args()

# -------------------------------------
# Load configuration
# -------------------------------------

object_name = args.object
cfg_fname = f'configs/{object_name}.yaml'
cfg.merge_from_file(cfg_fname)

if torch.cuda.is_available():
    DEVICE = torch.device('cuda', args.device)
else:
    DEVICE = torch.device('cpu')

# -------------------------------------
# Dataset & DataLoader
# -------------------------------------

num_frames = args.num_frames
num_samples = args.batch_size // num_frames

dataset = PACNeRFDataset(
    dataroot="dataset/pac-nerf",
    instance=object_name,
    verbose=True,
    max_frames=num_frames
)
loader = DataLoader(
    dataset,
    batch_size=num_frames,
    shuffle=True,
)

# -------------------------------------
# PINN Model
# -------------------------------------

solver = NavierCauchy(
    hid_dim=128,
    depth=8,
    model_type=mlp_dict[args.mlp],
    activation = nn.Tanh,
    
    # Environment
    ground_pos = 0,                    # The y-coord of the ground
    up_index   = dataset.up_index,     # The y axis will be the gravity direction
    gravity    = 9.8,                  # The gravitational acceleration

    # Physical parameters
    density = cfg.ELASTOMER.DENSITY,    # kg m⁻³
    youngs  = cfg.ELASTOMER.YOUNGS,     # Pa
    poissons= 0.1,  # cfg.ELASTOMER.POISSONS,
    
    # Physical parameter optimization options
    optimize_density    = False,        # Indicates whether optimize ρ
    optimize_youngs     = False,         # Indicates whether optimize E
    optimize_poissons   = True,        # Indicates whether optimize ν
).to(DEVICE)

generator = torch.Generator(device=DEVICE)
generator.manual_seed(1000)
solver.model.initialize_weights(
    generator=generator,
    init_fn=MLP_INIT_DICT[args.mlp_init],
)

# -------------------------------------
# Optimizers & Schedulers
# -------------------------------------

n_epochs = args.epochs

optimizers = [
    optim.AdamW(
        solver.network_parameters(),
        lr=args.learning_rate,
    ), 
    optim.Adam(
        solver.property_parameters(),
        lr=args.property_learning_rate,
    ), 
]

schedulers = [
    torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=n_epochs,
        eta_min=0,
    ) for optimizer in optimizers
]

# -------------------------------------
# Logging
# -------------------------------------

# Logger
loss_history = Averager()
loss_history_detailed = Averager()
prop_history = Averager()
prop_history.push({
    'density': solver.density,
    'youngs': solver.youngs,
    'poissons': solver.poissons,
}, flush=True)
tensor_trace = Averager()
lr_history = Averager()
ckpt_writer = CheckpointWriter(
    dir_name=f'./output/{object_name}_{args.tag}' if args.tag else f'./output/{object_name}',
    save_first=False,
    save_every=args.save_every,
    save_best=True,
    save_last=True,
    larger_better=False,
    overwrite=args.overwrite,
)
ckpt_writer.copy_code(__file__)
ckpt_writer.copy_code(inspect.getfile(NavierCauchy))
ckpt_writer.copy_code(
    inspect.getfile(NavierCauchy.__base__),
    'mlp.py',
)

# -------------------------------------
# Training Loop
# -------------------------------------

# The loss weights
loss_weight_pde: float = cfg.ELASTOMER.LOSS.PDE
loss_weight_gt: float = cfg.ELASTOMER.LOSS.GT
loss_weight_vel: float = cfg.ELASTOMER.LOSS.VEL
loss_weight_bc: float = cfg.ELASTOMER.LOSS.BC
loss_weight_ic: float = cfg.ELASTOMER.LOSS.IC

# The training loop
for epoch in range(n_epochs):
    
    for sample in tqdm(loader, desc=f"Epoch {epoch+1}/{n_epochs}"):
        
        # input preparation
        geometry: torch.Tensor = sample['geometry']
        displacement: torch.Tensor = sample['displacement']
        time_value: torch.Tensor = sample['time'][:, 0].reshape(-1, 1, 1)
        num_timesteps, num_points, _ = geometry.shape
        
        sample_indices = torch.randperm(num_points - 1)[:num_samples]
        num_points = len(sample_indices)
        
        geometry = geometry[:, sample_indices].to(DEVICE)  # ...........| T, P, 3
        displacement = displacement[:, sample_indices].to(DEVICE)  # ...| T, P, 3
        time_value = time_value.expand_as(geometry[..., 0:1])  # .......| T, P, 1
        time_value = time_value.to(DEVICE)
        xyzt = torch.cat([geometry, time_value], dim=-1)  # ............| T, P, 4
        
        if solver.model.input_shape == 'flat':
            geometry = geometry.flatten(0, 1)  # .......................| TP, 3
            displacement = displacement.flatten(0, 1)  # ...............| TP, 3 
            time_value = time_value.flatten(0, 1)  # ...................| TP, 1 
            xyzt = xyzt.flatten(0, 1)  # ...............................| TP, 4
        elif solver.model.input_shape == 'spatio-temporal':
            pass
        else:
            raise NotImplementedError(solver.model.input_shape)

        # forward
        losses, traces = solver.compute_loss(
            xyzt,
            time_dim=num_timesteps,
            point_dim=num_points,
            time=time_value,
            displacement=displacement,
            use_pde=True,
            use_vel=False,
            return_traces=True
        )
        traces: dict[str, torch.Tensor]
        losses: dict[str, torch.Tensor] = {
            'pde_loss': losses['pde_loss'] * loss_weight_pde,
            'gt_loss': losses['gt_loss'] * loss_weight_gt,
            'vel_loss': losses['vel_loss'] * loss_weight_vel,
            'bc_loss': losses['bc_loss'] * loss_weight_bc,
            'ic_loss': losses['ic_loss'] * loss_weight_ic,
        }
        
        # loss backward and gradient steps
        for val in traces.values():
            val.retain_grad()
        
        total_loss = sum(losses.values())
        total_loss.backward(retain_graph=True)
        
        tensor_trace.push({f'fwd_{key}': val.data for key, val in traces.items()})
        tensor_trace.push({f'bwd_{key}': val.grad for key, val in traces.items()})

        # torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) # clipping 
        for optimizer in optimizers:
            optimizer.step()
            optimizer.zero_grad()

        # logging the losses
        loss_history_detailed.push(losses)
        gc.collect()

    # logging the losses 
    epoch_loss_detailed = loss_history_detailed.flush()
    epoch_loss = sum(epoch_loss_detailed.values())

    # # scheduler step w.r.t. the total loss
    # for scheduler in schedulers:
    #     scheduler.step()

    # logging the physical parameters and total loss
    loss_history.push({
        'loss': epoch_loss,
    }, flush=True)
    prop_history.push({
        'youngs': solver.youngs.data,
        'poissons': solver.poissons.data,
        'density': solver.density.data,
    }, flush=True)
    lr_history.push({
        'network': optimizers[0].param_groups[0]['lr'],
        'prop': optimizers[1].param_groups[0]['lr'],
    }, flush=True)
    tensor_trace.flush()
    
    # save the checkpoints to the file(s)
    ckpt_writer.write({
        'epoch': epoch + 1,
        'model': {
            key: val.clone().detach().cpu()
            for key, val in solver.state_dict().items()
        },
        'model_config': solver.config,
        'optimizers': [
            optimizer.state_dict() 
            for optimizer in optimizers
        ],
        'loss_list': loss_history.gather(),
        'loss_detailed': loss_history_detailed.gather(),
        'lr_list': lr_history.gather(),
        'prop_traj': prop_history.gather(),
        'tensor_trace': tensor_trace.take(-1),
    }, score=epoch_loss)

    print(
        f"Epoch {epoch+1}", 
        f"Loss: {epoch_loss.item():.2f}", 
        f"LR: {lr_history.gather()['network'][-1]:.2E}",
        f"ρ: {solver.density.item():.4E}", 
        f"ν: {solver.poissons.item():.4E}", 
        f"E: {solver.youngs.item():.4E}", 
        sep=' ',
    )
