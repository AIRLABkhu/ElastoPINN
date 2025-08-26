import torch
from torch import nn, optim
from torch.utils.data import DataLoader
import torch.multiprocessing as mp


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
# Worker 함수 정의 (각 GPU에서 실행될 부분)
# -------------------------------------

def inner_loop_worker(proc_id, meta_state_dict, args):
    """
    하나의 GPU 프로세스에서 내부 루프 학습을 담당하는 함수입니다.
    """
    # 1. 장치 설정 및 데이터 로더 재생성
    DEVICE = torch.device(f'cuda:{proc_id}')
    torch.cuda.set_device(DEVICE)

    # 각 프로세스에서 데이터셋과 로더를 새로 생성하여 충돌 방지
    dataset = PACNeRFDataset(
        dataroot="dataset/pac-nerf",
        instance=args.object,
        verbose=False,  # 메인 프로세스에서만 로그를 출력하도록 설정
        max_frames=args.num_frames
    )
    loader = DataLoader(dataset, batch_size=args.num_frames, shuffle=True)
    num_samples = args.batch_size // args.num_frames

    # 2. 모델 생성 및 상태 복원, Perturbation 추가
    # create_solver_model 함수는 글로벌 cfg를 사용합니다.
    solver_model = NavierCauchy(
        hid_dim=128, 
        depth=8, 
        model_type=mlp_dict[args.mlp], 
        activation=nn.Tanh,
        ground_pos=0.0, 
        up_index=dataset.up_index, 
        gravity=9.8,
        density=cfg.ELASTOMER.DENSITY, 
        youngs=cfg.ELASTOMER.YOUNGS,
        poissons=cfg.ELASTOMER.POISSONS, 
        optimize_density=False,
        optimize_youngs=False, 
        optimize_poissons=True
    ).to(DEVICE)
    
    solver_model.load_state_dict(meta_state_dict)

    with torch.no_grad():
        for param in solver_model.parameters():
            perturbation = torch.randn_like(param) * args.perturbation
            param.add_(perturbation)

    # 3. 내부 학습 루프 실행
    print(f"[GPU {proc_id}] Starting Inner Loop Training...")
    
    optimizers = [
        optim.AdamW(solver_model.network_parameters(), lr=args.learning_rate),
        optim.AdamW(solver_model.property_parameters(), lr=args.property_learning_rate),
    ]
    schedulers = [
        torch.optim.lr_scheduler.CosineAnnealingLR(optimizers[0], T_max=args.inner_epochs, eta_min=0),
        torch.optim.lr_scheduler.CosineAnnealingLR(optimizers[1], T_max=args.inner_epochs - args.warmup, eta_min=0),
    ]
    
    select_lw = lambda arg_, cfg_: cfg_ if arg_ is None else arg_
    loss_weight_pde: float = select_lw(args.loss_pde, cfg.ELASTOMER.LOSS.PDE)
    loss_weight_gt: float = select_lw(args.loss_gt, cfg.ELASTOMER.LOSS.GT)
    loss_weight_ic: float = select_lw(args.loss_ic, cfg.ELASTOMER.LOSS.IC)
    loss_weight_bc: float = select_lw(args.loss_bc, cfg.ELASTOMER.LOSS.BC)

    for epoch in range(args.inner_epochs):
        # tqdm은 메인 프로세스에서만 사용하도록 proc_id==0일 때만 활성화
        loop_desc = f"GPU {proc_id} Inner Epoch {epoch+1}/{args.inner_epochs}"
        pbar = tqdm(loader, desc=loop_desc, leave=False) if proc_id == 0 else loader

        for sample in pbar:
            b_warmup_done = epoch >= args.warmup
            
            if isinstance(solver_model.model, LoRAMLP):
                if b_warmup_done: solver_model.model.lora()
                else: solver_model.model.linear()
            
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
                geometry, displacement, time_value, xyzt = [t.flatten(0, 1) for t in (geometry, displacement, time_value, xyzt)]
            
            losses = solver_model.compute_loss(
                xyzt, time_dim=num_timesteps, 
                point_dim=num_points, 
                time=time_value,
                displacement=displacement, 
                use_pde=b_warmup_done,
                use_ic=True, 
                use_bc=True, 
                use_vel=False,
            )
            losses: dict[str, torch.Tensor] = {
                'pde_loss': losses['pde_loss'] * loss_weight_pde,
                'gt_loss': losses['gt_loss'] * loss_weight_gt,
                'bc_loss': losses['bc_loss'] * loss_weight_bc,
                'ic_loss': losses['ic_loss'] * loss_weight_ic,
            }
            
            total_loss = sum(loss_val for name, loss_val in losses.items() if name != 'cp_loss')
            total_loss.backward()

            optimizers[0].step(); optimizers[0].zero_grad()
            if b_warmup_done:
                optimizers[1].step(); optimizers[1].zero_grad()

    print(f"[GPU {proc_id}] Inner Loop Training Finished.")
    
    # 4. 결과 반환 (CPU로 이동)
    cpu_state_dict = {k: v.cpu() for k, v in solver_model.state_dict().items()}
    
    # losses 딕셔너리의 각 텐서에서 계산 기록을 분리(.detach())하고 CPU로 옮깁니다.
    final_losses = {name: val.detach().cpu() for name, val in losses.items()}
    return cpu_state_dict, final_losses


def main():
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
    add('--num-inner-models', '-s', type=int, default=4) 
    add('--outer-epochs', '-oe', type=int, default=10) 
    add('--inner-epochs', '-ie', type=int, default=1000)
    add('--perturbation', '-p', type=float, default=3.0E-2)
    args = parser.parse_args()

    object_name = args.object
    cfg_fname = f'configs/{object_name}.yaml'
    cfg.merge_from_file(cfg_fname)

    if torch.cuda.is_available():
        MAIN_DEVICE = torch.device('cuda', args.device)
    else:
        MAIN_DEVICE = torch.device('cpu')

    # s를 4로 고정하고, 사용 가능한 GPU 수 확인
    s = 8
    args.num_inner_models = s
    if torch.cuda.device_count() < s:
        raise ValueError(f"This script requires {s} GPUs, but only {torch.cuda.device_count()} are available.")


    # -------------------------------------
    # PINN Model (Meta Model)
    # -------------------------------------
    def create_solver_model(initialize_poissons = cfg.ELASTOMER.POISSONS):
        solver = NavierCauchy(
            hid_dim=128, 
            depth=8, 
            model_type=mlp_dict[args.mlp], 
            activation=nn.Tanh,
            ground_pos=0.0, 
            up_index=1, 
            gravity=9.8,
            density=cfg.ELASTOMER.DENSITY, 
            youngs=cfg.ELASTOMER.YOUNGS,
            poissons=initialize_poissons, 
            optimize_density=False,
            optimize_youngs=False, 
            optimize_poissons=True
        ).to(MAIN_DEVICE)
        return solver

    meta_model = create_solver_model()

    # -------------------------------------
    # Logging
    # -------------------------------------
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
    # ... (기존 로깅 코드)

    loss_history = Averager()
    loss_history_detailed = Averager()
    prop_history = Averager()
    prop_history.push({
        'density': cfg.ELASTOMER.DENSITY, 
        'youngs': cfg.ELASTOMER.YOUNGS,
        'poissons': cfg.ELASTOMER.POISSONS,
    }, flush=True)

    # -------------------------------------
    # Outer Training Loop
    # -------------------------------------
    print("="*45)
    print("             STARTING OUTER LOOP")
    print("="*45)

    

    for outer_epoch in range(args.outer_epochs):
        print(f"\n{'='*20} Outer Epoch {outer_epoch+1}/{args.outer_epochs} {'='*20}")

        meta_model.cpu() # state_dict를 복사하기 전에 CPU로 이동
        meta_model_state_dict = meta_model.state_dict()

        # 멀티프로세싱 풀 생성 및 실행
        print(f"\n--- Starting parallel training on {s} GPUs for inner models ---")
        pool_args = [(i, meta_model_state_dict, args) for i in range(s)]
        
        with mp.Pool(processes=s) as pool:
            trained_results = pool.starmap(inner_loop_worker, pool_args)
        
        trained_state_dicts, trained_losses_list = zip(*trained_results)

        # 학습된 모델의 속성(poissons)을 얻기 위해 임시 모델 생성 및 상태 로드
        trained_poissons = []
        temp_models = [create_solver_model().cpu() for _ in range(s)]
        for model, state_dict in zip(temp_models, trained_state_dicts):
            model.load_state_dict(state_dict)
            trained_poissons.append(model.poissons.data)

        # --- 결과 평균화 
        print("\n--- Averaging model parameters ---")
        combined_state_dict = copy.deepcopy(trained_state_dicts[0])
        for state_dict in trained_state_dicts[1:]:
            for key in combined_state_dict.keys():
                combined_state_dict[key] += state_dict[key]
        for key in combined_state_dict.keys():
            combined_state_dict[key] /= s

        print("\n--- Averaging model poisson's ratios ---")
        avg_trained_poissons = torch.stack(trained_poissons).mean()
        print(f"\n current avg poissons: {avg_trained_poissons.item()}")

        print("\n--- Averaging loss values ---")
        avg_losses = {
            key: sum(d[key].item() for d in trained_losses_list) / s
            for key in trained_losses_list[0].keys()
        }
        for loss_name, avg_value in avg_losses.items():
            print(f"Average {loss_name}: {avg_value:.6f}")

        # 평균화된 값으로 새로운 meta_model 업데이트
        meta_model = create_solver_model(initialize_poissons=avg_trained_poissons).to(MAIN_DEVICE)
        meta_model.load_state_dict(combined_state_dict)
        print("--- New meta_model created from averaged results ---")

        #
        loss_history_detailed.push(avg_losses)
        total_avg_loss = sum(avg_losses.values())
        loss_history.push({'total_loss': total_avg_loss})
        prop_history.push({
            'density': meta_model.density, 
            'youngs': meta_model.youngs,
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
    mp.set_start_method('spawn', force=True)
    main()