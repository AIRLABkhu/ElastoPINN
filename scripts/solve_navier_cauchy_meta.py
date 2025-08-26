from argparse import ArgumentParser
from tqdm import tqdm
import inspect

import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from configs.config import CFG as cfg
from data.pac_nerf import PACNeRFDataset

import torch.nn.functional as F
from collections import OrderedDict

from models.mlp import mlp_dict, LoRAMLP
from models.navier_cauchy_neo_hookean import NavierCauchy
# from models.navier_cauchy_neo_hookean_piola import NavierCauchy
from utils.nn import SIREN
from utils.logging import (
    Averager,
    CheckpointWriter,
)
import copy


# -------------------------------------
# Load configuration
# -------------------------------------

parser = ArgumentParser('Navier-Cauchy')
add = parser.add_argument
add('--device', '-d', type=int, default=0)
add('--object', '-o', type=str.lower, default='bird', choices=PACNeRFDataset.INSTANCES)
add('--mlp', '-mlp', type=str.lower, default='mlp', choices=mlp_dict.keys())
add('--num-frames', '-nf', type=int, default=14)
add('--batch-size', '-bs', type=int, default=20000)
add('--learning-rate', '-lr', type=float, default=1.0E-4)
add('--property-learning-rate', '-plr', type=float, default=1.0E-1)
add('--loss-pde', type=float, default=None)
add('--loss-gt', type=float, default=None)
add('--loss-ic', type=float, default=None)
add('--loss-bc', type=float, default=None)
add('--epochs', '-e', type=int, default=10_000)
add('--warmup', '-w', type=int, default=5_000)
add('--save-every', type=int, default=1_000)
add('--tag', type=str, default=None)
add('--overwrite', action='store_true', default=False)
add('--const-lr', action='store_true', default=False)

## Meta learning 
add('--num-inner-models', '-s', type=int, default=4) 
add('--outer-epochs', '-oe', type=int, default=10) 
add('--inner-epochs', '-ie', type=int, default=1000)
add('--perturbation', '-p', type=float, default=1.0E-3)
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

# -------------------------------------
# PINN Model
# -------------------------------------
outer_loader = DataLoader(dataset, batch_size=len(dataset), shuffle=False)
outer_data_sample = next(iter(outer_loader))

loader = DataLoader(dataset, batch_size=num_frames, shuffle=True)


def create_solver_model(initialize_poissons = cfg.ELASTOMER.POISSONS):
    solver = NavierCauchy(
        hid_dim=128,
        depth=8,
        model_type=mlp_dict[args.mlp],
        activation=nn.Tanh,
        ground_pos = 0.0,          
        up_index   = dataset.up_index,
        gravity    = 9.8,
        density    = cfg.ELASTOMER.DENSITY,
        youngs     = cfg.ELASTOMER.YOUNGS,
        poissons   = initialize_poissons,
        optimize_density    = False,
        optimize_youngs     = False,
        optimize_poissons   = True,
    ).to(DEVICE)
    return solver


def run_inner_training_loop(solver_model, inner_epochs, n_warmups):
    print(f"\n--- Starting Inner Loop Training for one model ({inner_epochs} epochs) ---")
    
    # 모델별로 옵티마이저와 스케줄러를 새로 생성
    optimizers = [
        optim.AdamW(solver_model.network_parameters(), lr=args.learning_rate), 
        optim.AdamW(solver_model.property_parameters(), lr=args.property_learning_rate), 
    ]
    schedulers = [
        torch.optim.lr_scheduler.CosineAnnealingLR(optimizers[0], T_max=inner_epochs, eta_min=0), 
        torch.optim.lr_scheduler.CosineAnnealingLR(optimizers[1], T_max=inner_epochs - n_warmups, eta_min=0), 
    ]
    
    # The loss weights
    select_lw = lambda arg_, cfg_: cfg_ if arg_ is None else arg_
    loss_weight_pde: float = select_lw(args.loss_pde, cfg.ELASTOMER.LOSS.PDE)
    loss_weight_gt: float = select_lw(args.loss_gt, cfg.ELASTOMER.LOSS.GT)
    loss_weight_ic: float = select_lw(args.loss_ic, cfg.ELASTOMER.LOSS.IC)
    loss_weight_bc: float = select_lw(args.loss_bc, cfg.ELASTOMER.LOSS.BC)

    for epoch in range(inner_epochs):
        for sample in tqdm(loader, desc=f"Inner Epoch {epoch+1}/{inner_epochs}", leave=False):
            b_warmup_done = epoch >= n_warmups
            
            if isinstance(solver_model.model, LoRAMLP):
                if b_warmup_done: solver_model.model.lora()
                else: solver_model.model.linear()
            
            # input preparation (기존과 동일)
            geometry: torch.Tensor = sample['geometry']
            displacement: torch.Tensor = sample['displacement']
            time_value: torch.Tensor = sample['time'][:, 0].reshape(-1, 1, 1)
            num_timesteps, num_points, _ = geometry.shape
            
            sample_indices = torch.randperm(num_points - 1)[:num_samples]
            num_points = len(sample_indices)
            
            geometry = geometry[:, sample_indices].to(DEVICE)
            displacement = displacement[:, sample_indices].to(DEVICE)
            time_value = time_value.expand_as(geometry[..., 0:1]).to(DEVICE)
            xyzt = torch.cat([geometry, time_value], dim=-1)
            
            if solver_model.model.input_shape == 'flat':
                geometry = geometry.flatten(0, 1)
                displacement = displacement.flatten(0, 1)
                time_value = time_value.flatten(0, 1)
                xyzt = xyzt.flatten(0, 1)
            
            # forward (기존과 동일)
            losses = solver_model.compute_loss(
                xyzt,
                time_dim=num_timesteps,
                point_dim=num_points,
                time=time_value,
                displacement=displacement,
                use_pde=b_warmup_done,
                use_ic=True, use_bc=True, use_vel=False,
            )
            losses: dict[str, torch.Tensor] = {
                'pde_loss': losses['pde_loss'] * loss_weight_pde,
                'gt_loss': losses['gt_loss'] * loss_weight_gt,
                'bc_loss': losses['bc_loss'] * loss_weight_bc,
                'ic_loss': losses['ic_loss'] * loss_weight_ic,
            }
            
            # backward and step (기존과 동일)
            total_loss = sum(loss_val for name, loss_val in losses.items() if name != 'cp_loss')
            total_loss.backward()

            # torch.nn.utils.clip_grad_norm_(solver_model.parameters(), 1.0)
            optimizers[0].step()
            optimizers[0].zero_grad()
            if b_warmup_done:
                optimizers[1].step()
                optimizers[1].zero_grad()


    print("--- Inner Loop Training Finished ---")
    return solver_model, losses



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


loss_history = Averager()
loss_history_detailed = Averager()
prop_history = Averager()
prop_history.push({
    'density': cfg.ELASTOMER.DENSITY,
    'youngs': cfg.ELASTOMER.YOUNGS,
    'poissons': cfg.ELASTOMER.POISSONS,
}, flush=True)

lr_history = Averager()
lame_history = Averager()


print("=============================================")
print("         STARTING OUTER LOOP"                 )
print("=============================================")

# s개의 모델 인스턴스 생성
s = args.num_inner_models

# 로깅
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


meta_model = create_solver_model()
for outer_epoch in range(args.outer_epochs):
    print(f"\n{'='*20} Outer Epoch {outer_epoch+1}/{args.outer_epochs} {'='*20}")

    print(f"\n--- Creating {s} inner models from meta_model ---")
    meta_model_state_dict = meta_model.state_dict()
    inner_models = [create_solver_model() for _ in range(s)]
    for model in inner_models:
        model.load_state_dict(meta_model_state_dict)
    
     
    print(f"\n--- Adding perturbation (magnitude: {args.perturbation}) to inner models ---")
    with torch.no_grad():
        for model in inner_models:
            for param in model.parameters():
                perturbation = torch.randn_like(param) * args.perturbation
                param.add_(perturbation)

    trained_results = [
        run_inner_training_loop(model, args.inner_epochs, args.warmup) for model in inner_models
    ]
    trained_models, trained_losses_list = zip(*trained_results)

    trained_state_dicts = [model.state_dict() for model in trained_models]
    trained_poissons = [model.poissons.data for model in trained_models] 


    print("\n--- Averaging model parameters ---")
    combined_state_dict = copy.deepcopy(trained_state_dicts[0])
    for state_dict in trained_state_dicts[1:]:
        for key in combined_state_dict.keys():
            combined_state_dict[key] += state_dict[key]

    for key in combined_state_dict.keys():
        combined_state_dict[key] /= s


    print("\n--- Averaging model poisson's ratios ---")
    avg_trained_poissons = torch.stack(trained_poissons).mean()
    print("\n current avg poissons: ", avg_trained_poissons)

    print("\n--- Averaging loss values ---")
    avg_losses = {
        key: sum(d[key].item() for d in trained_losses_list) / s
        for key in trained_losses_list[0].keys()
    }
    for loss_name, avg_value in avg_losses.items():
        print(f"Average {loss_name}: {avg_value:.6f}")

   
    combined_model = create_solver_model(initialize_poissons=avg_trained_poissons)
    combined_model.load_state_dict(combined_state_dict) 
    meta_model = combined_model
    print("--- New combined_model created and set as meta_model for the next outer epoch ---")


    print("--- Pushing averaged values to history ---")
    loss_history_detailed.push(avg_losses)

    total_avg_loss = sum(avg_losses.values())
    loss_history.push({'total_loss': total_avg_loss})
    prop_history.push({
        'density': meta_model.density,
        'youngs': meta_model.youngs,
        'poissons': meta_model.poissons, # 이 값은 avg_trained_poissons와 동일
    }, flush=True)


    print("--- Saving checkpoint ---")
    ckpt_writer.write({
        'args': vars(args),
        'epoch': outer_epoch + 1, # outer_epoch를 사용
        'model': {
            key: val.clone().detach().cpu()
            for key, val in meta_model.state_dict().items() 
        },
        'model_config': meta_model.config, # meta_model의 config 사용
        'optimizers': [], # outer loop optimizer가 없으므로 비워두거나 해당 optimizer 추가
        'loss_list': loss_history.gather(),
        'loss_detailed': loss_history_detailed.gather(),
        'prop_traj': prop_history.gather(),
        # 'lame_traj': lame_history.gather(), # lame_history가 있다면 이 부분도 업데이트
    }, score=total_avg_loss) # 현재 epoch의 평균 손실을 점수로 사용