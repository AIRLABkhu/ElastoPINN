import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.nn import parallel # Import the parallel module

from argparse import ArgumentParser
from tqdm import tqdm
import copy

from configs.config import CFG as cfg
from data.pac_nerf import PACNeRFDataset
from models.mlp import mlp_dict, LoRAMLP
from models.navier_cauchy_neo_hookean import NavierCauchy
from utils.logging import (
    Averager,
    CheckpointWriter,
)

# -------------------------------------
# Training Wrapper for parallel_apply
# -------------------------------------
class TrainingWrapper(nn.Module):
    """
    A wrapper module to handle a single training step (loss calculation and backward pass).
    This is necessary for use with `torch.nn.parallel.parallel_apply`.
    """
    def __init__(self, model, loss_weights):
        super().__init__()
        self.model = model
        self.loss_weights = loss_weights

    def forward(self, input_dict):
        # Extract state required for this specific step
        b_warmup_done = input_dict.pop('b_warmup_done')
        
        # Handle model-specific state changes (e.g., LoRA)
        if isinstance(self.model.model, LoRAMLP):
            if b_warmup_done: self.model.model.lora()
            else: self.model.model.linear()

        # Compute losses
        losses = self.model.compute_loss(**input_dict)
        weighted_losses = {
            'pde_loss': losses['pde_loss'] * self.loss_weights['pde'],
            'gt_loss': losses['gt_loss'] * self.loss_weights['gt'],
            'bc_loss': losses['bc_loss'] * self.loss_weights['bc'],
            'ic_loss': losses['ic_loss'] * self.loss_weights['ic'],
        }
        
        # Perform backward pass
        total_loss = sum(loss_val for name, loss_val in weighted_losses.items() if name != 'cp_loss')
        total_loss.backward()
        
        # Return detached losses for logging
        return {name: val.detach() for name, val in weighted_losses.items()}

def main():
    parser = ArgumentParser('Navier-Cauchy')
    # ... (Argument parsing remains the same)
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
    add('--num-inner-models', '-s', type=int, default=4) 
    add('--outer-epochs', '-oe', type=int, default=10) 
    add('--inner-epochs', '-ie', type=int, default=1000)
    add('--perturbation', '-p', type=float, default=1.0E-3)
    args = parser.parse_args()

    object_name = args.object
    cfg_fname = f'configs/{object_name}.yaml'
    cfg.merge_from_file(cfg_fname)

    # s를 4로 고정하고, 사용 가능한 GPU 수 확인 및 장치 설정
    s = 3
    args.num_inner_models = s
    if torch.cuda.device_count() < s:
        raise ValueError(f"This script requires {s} GPUs, but only {torch.cuda.device_count()} are available.")
    
    devices = [torch.device(f'cuda:{i}') for i in range(s)]
    MAIN_DEVICE = devices[args.device]
    torch.cuda.set_device(MAIN_DEVICE)
    
    # -------------------------------------
    # PINN Model (Meta Model)
    # -------------------------------------
    def create_solver_model(initialize_poissons=cfg.ELASTOMER.POISSONS):
        solver = NavierCauchy(
            hid_dim=128, depth=8, model_type=mlp_dict[args.mlp], 
            activation=nn.Tanh, ground_pos=0.0, up_index=1, gravity=9.8,
            density=cfg.ELASTOMER.DENSITY, youngs=cfg.ELASTOMER.YOUNGS,
            poissons=initialize_poissons, optimize_density=False,
            optimize_youngs=False, optimize_poissons=True
        ).to(MAIN_DEVICE)
        return solver

    meta_model = create_solver_model()

    # -------------------------------------
    # Logging
    # -------------------------------------
    # ... (Logging setup remains the same)
    ckpt_writer = CheckpointWriter(
        dir_name=f'./output/{object_name}_{args.tag}' if args.tag else f'./output/{object_name}',
        save_first=False, save_every=args.save_every, save_best=True,
        save_last=True, larger_better=False, overwrite=args.overwrite,
    )
    ckpt_writer.copy_code(__file__)
    loss_history = Averager()
    loss_history_detailed = Averager()
    prop_history = Averager()
    prop_history.push({
        'density': cfg.ELASTOMER.DENSITY, 
        'youngs': cfg.ELASTOMER.YOUNGS,
        'poissons': cfg.ELASTOMER.POISSONS,
    }, flush=True)

    # -------------------------------------
    # Dataset and Loss Weights (defined once)
    # -------------------------------------
    dataset = PACNeRFDataset(
        dataroot="dataset/pac-nerf", instance=args.object,
        verbose=True, max_frames=args.num_frames
    )
    loader = DataLoader(dataset, batch_size=args.num_frames, shuffle=True)
    num_samples = args.batch_size // args.num_frames

    select_lw = lambda arg_, cfg_: cfg_ if arg_ is None else arg_
    loss_weights = {
        'pde': select_lw(args.loss_pde, cfg.ELASTOMER.LOSS.PDE),
        'gt': select_lw(args.loss_gt, cfg.ELASTOMER.LOSS.GT),
        'ic': select_lw(args.loss_ic, cfg.ELASTOMER.LOSS.IC),
        'bc': select_lw(args.loss_bc, cfg.ELASTOMER.LOSS.BC),
    }

    # -------------------------------------
    # Outer Training Loop
    # -------------------------------------
    print("="*45); print("             STARTING OUTER LOOP"); print("="*45)

    for outer_epoch in range(args.outer_epochs):
        print(f"\n{'='*20} Outer Epoch {outer_epoch+1}/{args.outer_epochs} {'='*20}")

        # 1. Replicate the meta-model to all specified devices
        print(f"\n--- Replicating meta-model to {s} GPUs ---")
        replicas = parallel.replicate(meta_model, devices)

        # 2. Apply perturbation to each replica
        with torch.no_grad():
            for replica in replicas:
                for param in replica.parameters():
                    perturbation = torch.randn_like(param) * args.perturbation
                    param.add_(perturbation)
        
        # 3. Create optimizers, schedulers, and training wrappers for each replica
        optimizers_list, schedulers_list, training_wrappers = [], [], []
        for replica in replicas:
            opts = [
                optim.AdamW(replica.network_parameters(), lr=args.learning_rate),
                optim.AdamW(replica.property_parameters(), lr=args.property_learning_rate),
            ]
            scheds = [
                torch.optim.lr_scheduler.CosineAnnealingLR(opts[0], T_max=args.inner_epochs, eta_min=0),
                torch.optim.lr_scheduler.CosineAnnealingLR(opts[1], T_max=args.inner_epochs - args.warmup, eta_min=0),
            ]
            optimizers_list.append(opts)
            schedulers_list.append(scheds)
            training_wrappers.append(TrainingWrapper(replica, loss_weights))

        # 4. Inner Training Loop
        print(f"--- Starting parallel inner loop training for {args.inner_epochs} epochs ---")
        pbar = tqdm(range(args.inner_epochs), desc=f"Outer Epoch {outer_epoch+1}")
        for epoch in pbar:
            for sample in loader: # This loop will run only once as batch_size == num_frames
                b_warmup_done = epoch >= args.warmup

                # Prepare inputs for each replica for parallel_apply
                parallel_inputs = []
                sample_indices = torch.randperm(sample['geometry'].shape[1] - 1)[:num_samples]
                num_points = len(sample_indices)
                num_timesteps = args.num_frames

                for device in devices:
                    geometry = sample['geometry'][:, sample_indices].to(device)
                    displacement = sample['displacement'][:, sample_indices].to(device)
                    time_value = sample['time'][:, 0].reshape(-1, 1, 1).expand_as(geometry[..., 0:1]).to(device)
                    xyzt = torch.cat([geometry, time_value], dim=-1)
                    
                    if meta_model.model.input_shape == 'flat':
                        geometry, displacement, time_value, xyzt = [t.flatten(0, 1) for t in (geometry, displacement, time_value, xyzt)]
                    
                    input_dict = {
                        'xyzt': xyzt, 'time_dim': num_timesteps, 'point_dim': num_points, 
                        'time': time_value, 'displacement': displacement, 
                        'use_pde': b_warmup_done, 'use_ic': True, 'use_bc': True, 'use_vel': False,
                        'b_warmup_done': b_warmup_done, # Pass epoch state to wrapper
                    }
                    parallel_inputs.append((input_dict,))

                # 5. Apply the training step in parallel
                # This executes the forward pass of TrainingWrapper on each GPU, which includes the backward pass
                last_batch_losses_parallel = parallel.parallel_apply(training_wrappers, parallel_inputs)

                # 6. Step optimizers for each replica
                for opts in optimizers_list:
                    opts[0].step(); opts[0].zero_grad()
                    if b_warmup_done:
                        opts[1].step(); opts[1].zero_grad()

            # Step schedulers after each epoch
            for scheds in schedulers_list:
                scheds[0].step()
                if b_warmup_done:
                    scheds[1].step()
        
        print("--- Inner Loop Training Finished. ---")
        
        # 7. Gather and Average Results
        print("\n--- Gathering and averaging results ---")
        # Gather losses from the last training step
        gathered_losses_by_name = {
            key: parallel.gather([d[key] for d in last_batch_losses_parallel], MAIN_DEVICE)
            for key in last_batch_losses_parallel[0].keys()
        }
        avg_losses = {key: val.mean().item() for key, val in gathered_losses_by_name.items()}

        for loss_name, avg_value in avg_losses.items():
            print(f"Average {loss_name}: {avg_value:.6f}")

        # Collect state dicts and properties from replicas
        trained_state_dicts = [{k: v.cpu() for k, v in r.state_dict().items()} for r in replicas]
        trained_poissons = [r.poissons.data.cpu() for r in replicas]
        
        # Average model parameters
        combined_state_dict = copy.deepcopy(trained_state_dicts[0])
        for state_dict in trained_state_dicts[1:]:
            for key in combined_state_dict.keys():
                combined_state_dict[key] += state_dict[key]
        for key in combined_state_dict.keys():
            combined_state_dict[key] /= s

        # Average physical properties
        avg_trained_poissons = torch.stack(trained_poissons).mean()
        print(f"\ncurrent avg poissons: {avg_trained_poissons.item()}")

        # 8. Update and save the meta-model
        meta_model = create_solver_model(initialize_poissons=avg_trained_poissons).to(MAIN_DEVICE)
        meta_model.load_state_dict(combined_state_dict)
        print("\n--- New meta_model created from averaged results ---")

        # Logging and checkpointing
        loss_history_detailed.push(avg_losses)
        total_avg_loss = sum(avg_losses.values())
        loss_history.push({'total_loss': total_avg_loss})
        prop_history.push({
            'density': meta_model.density, 'youngs': meta_model.youngs,
            'poissons': meta_model.poissons,
        }, flush=True)

        print("--- Saving checkpoint ---")
        ckpt_writer.write({
            'args': vars(args), 'epoch': outer_epoch + 1,
            'model': {k: v.cpu() for k, v in meta_model.state_dict().items()},
            'model_config': meta_model.config, 'optimizers': [],
            'loss_list': loss_history.gather(),
            'loss_detailed': loss_history_detailed.gather(),
            'prop_traj': prop_history.gather(),
        }, score=total_avg_loss)

if __name__ == '__main__':
    main()