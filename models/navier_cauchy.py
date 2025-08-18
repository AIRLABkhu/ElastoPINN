from typing import Type
import torch
from torch import nn, autograd
from .mlp import MLPBase, MLP
from .solver import Solver


class NavierCauchy(Solver):
    def __init__(
        self,
        # specifying the model architecture
        hid_dim: int = 200,
        depth: int = 6,
        activation: Type[nn.Module] = nn.Tanh,
        model_type: Type[MLPBase] = MLP,
        # configuring the environment
        ground_pos: float = 1.0E-6,
        gravity: float = 9.8,
        up_index: int = 1,
        # configuring the physical properties
        density: float = 1.0E+3,
        youngs: float = 1.0E+6,
        poissons: float = 3.0E-1,
        optimize_density : bool=False,
        optimize_youngs  : bool=False,
        optimize_poissons: bool=False,
        **kwargs,
    ):
        super().__init__(
            in_dim=4, 
            hid_dim=hid_dim, 
            out_dim=3, 
            depth=depth, 
            activation=activation, 
            model_type=model_type, 
            ground_pos=ground_pos, 
            gravity=gravity, 
            up_index=up_index, 
        )

        # Log parameterization for positive values
        self.register_physical_property('density', density, optimize_density)
        self.log_density = nn.Parameter(
            torch.log(torch.tensor(density)), 
            requires_grad=optimize_density, 
        )
        self.register_physical_property('youngs', youngs, optimize_youngs)
        self.log_youngs = nn.Parameter(
            torch.log(torch.tensor(youngs)), 
            requires_grad=optimize_youngs, 
        )
        self.register_physical_property('poissons', poissons, optimize_poissons)
        # self.logit_poissons = nn.Parameter(
        #     torch.logit(torch.tensor(poissons * 2.0)), 
        #     requires_grad=optimize_poissons, 
        # )
        self.log_poissons = nn.Parameter(
            torch.log(torch.tensor(poissons)), 
            requires_grad=optimize_poissons, 
        )

    # Physical properties
    @property
    def density(self): 
        return torch.exp(self.log_density)

    @property
    def youngs(self): 
        return torch.exp(self.log_youngs)

    @property
    def poissons(self): 
        return torch.sigmoid(self.log_poissons) 

    def property_parameters(self):
        yield self.log_density
        yield self.log_youngs
        yield self.logit_poissons

    def compute_loss(
        self,
        xyzt: torch.Tensor,
        time_dim: int, point_dim: int,
        displacement: torch.Tensor = None,
        time: torch.Tensor = None,
        f_ext: torch.Tensor = None,
        use_pde: bool = True,
        use_ic: bool = True,
        use_bc: bool = True,
        use_vel: bool = False,
    ) -> dict[str, torch.Tensor]:
        """ Calculates the loss for the Navier-Cauchy equation. """
        
        kwargs = {'device': xyzt.device, 'dtype': xyzt.dtype}
        xyzt.requires_grad_(True)
        
        # 1. MLP forward
        mlp_output = self.forward(xyzt)
        uvw_local = mlp_output.local_branch
        uvw_global = mlp_output.global_branch
        u, v, w = torch.unbind(uvw_local, dim=-1)

        # 2. autograds (u, v, w) with respect to xyzt.
        grad_u = autograd.grad(
            outputs=u, inputs=xyzt, 
            grad_outputs=torch.ones_like(u), 
            create_graph=True,
        )[0]
        du_dx, du_dy, du_dz, du_dt = torch.unbind(grad_u, dim=-1)

        grad_v = autograd.grad(
            outputs=v, inputs=xyzt, 
            grad_outputs=torch.ones_like(v), 
            create_graph=True,
        )[0]
        dv_dx, dv_dy, dv_dz, dv_dt = torch.unbind(grad_v, dim=-1)

        grad_w = autograd.grad(
            outputs=w, inputs=xyzt, 
            grad_outputs=torch.ones_like(w), 
            create_graph=True,
        )[0]
        dw_dx, dw_dy, dw_dz, dw_dt = torch.unbind(grad_w, dim=-1)
        
        du_dt2 = autograd.grad(
            outputs=du_dt, inputs=xyzt, 
            grad_outputs=torch.ones_like(du_dt), 
            create_graph=True,
        )[0][..., 3]
        dv_dt2 = autograd.grad(
            outputs=dv_dt, inputs=xyzt, 
            grad_outputs=torch.ones_like(dv_dt), 
            create_graph=True,
        )[0][..., 3]
        dw_dt2 = autograd.grad(
            outputs=dw_dt, inputs=xyzt, 
            grad_outputs=torch.ones_like(dw_dt), 
            create_graph=True,
        )[0][..., 3]

        # 3. PDE Loss (residual of the Navier-Cauchy equation)
        pde_loss = torch.tensor(0.0, **kwargs)
        if use_pde:
            lmbda = (self.youngs * self.poissons) / ((1 + self.poissons) * (1 - 2 * self.poissons))
            mu = self.youngs / (2 * (1 + self.poissons))

            # Deformation gradient and Neo-Hookean stress tensor
            grad_u_mat = torch.stack([
                torch.stack([du_dx, du_dy, du_dz], dim=-1),
                torch.stack([dv_dx, dv_dy, dv_dz], dim=-1),
                torch.stack([dw_dx, dw_dy, dw_dz], dim=-1),
            ], dim=-2)
            I = torch.eye(3, **kwargs).unsqueeze(0)
            F = I + grad_u_mat
            J = torch.det(F)

            j_threshold = 1e-4 
            reg_loss = torch.mean(torch.relu(j_threshold - J)**2)

            F_stable = F + torch.eye(3, **kwargs) * 1e-9
            try:
                F_inv_T = torch.inverse(F_stable).transpose(-1, -2)
            except torch.linalg.LinAlgError:
                # 만약 역행렬 계산이 실패하면 (매우 드묾), 학습을 중단시키지 않고
                # 단위 행렬을 사용하여 해당 스텝의 그래디언트를 약화시킵니다.
                F_inv_T = torch.eye(3, **kwargs).unsqueeze(0)

            j_threshold = 1e-4  # 예: 0.0001 (부피가 99.99% 이상 줄어들지 않도록 함)
            log_J = torch.log(torch.clamp(J, min=j_threshold))

            # origin 
            # sigma = mu * (B - I) * inv_J + lmbda * log_J * inv_J * I 

            # 1st piola kirchhoff stress tensor
            # P = μF + (λlogJ - μ)F⁻ᵀ
            P = mu * F + (lmbda * log_J[..., None, None] - mu) * F_inv_T

            # Divergence of stress
            sigma_xx, sigma_xy, sigma_xz = P[..., 0, 0], P[..., 0, 1], P[..., 0, 2]
            sigma_yx, sigma_yy, sigma_yz = P[..., 1, 0], P[..., 1, 1], P[..., 1, 2]
            sigma_zx, sigma_zy, sigma_zz = P[..., 2, 0], P[..., 2, 1], P[..., 2, 2]

            div_sigma_x = (
                autograd.grad(sigma_xx, xyzt, torch.ones_like(sigma_xx), create_graph=True)[0][..., 0] + 
                autograd.grad(sigma_xy, xyzt, torch.ones_like(sigma_xy), create_graph=True)[0][..., 1] + 
                autograd.grad(sigma_xz, xyzt, torch.ones_like(sigma_xz), create_graph=True)[0][..., 2]
            )
            div_sigma_y = (
                autograd.grad(sigma_yx, xyzt, torch.ones_like(sigma_yx), create_graph=True)[0][..., 0] + 
                autograd.grad(sigma_yy, xyzt, torch.ones_like(sigma_yy), create_graph=True)[0][..., 1] + 
                autograd.grad(sigma_yz, xyzt, torch.ones_like(sigma_yz), create_graph=True)[0][..., 2]
            )
            div_sigma_z = (
                autograd.grad(sigma_zx, xyzt, torch.ones_like(sigma_zx), create_graph=True)[0][..., 0] + 
                autograd.grad(sigma_zy, xyzt, torch.ones_like(sigma_zy), create_graph=True)[0][..., 1] + 
                autograd.grad(sigma_zz, xyzt, torch.ones_like(sigma_zz), create_graph=True)[0][..., 2]
            )

            # Residual of the momentum equation
            pde_x = self.density * du_dt2 - div_sigma_x
            pde_y = self.density * dv_dt2 - div_sigma_y + self.density * self.gravity
            pde_z = self.density * dw_dt2 - div_sigma_z
            
            ## 
            # incompressilibility_weight = 1E+4
            # incompressilibility_loss = torch.mean(J-1) * incompressilibility_weight
            # cp_loss = incompressilibility_loss
            pde_loss = torch.mean(pde_x**2 + pde_y**2 + pde_z**2)  + reg_loss

        # 4. Initial Condition (IC) Loss
        ic_loss = torch.tensor(0.0, **kwargs)
        if use_ic :
            ic_mask = xyzt[..., 3] == 0.0
            if torch.any(ic_mask):
                # 1. 초기 변위 손실 (Initial Displacement Loss)
                u_ic = u[ic_mask]
                v_ic = v[ic_mask]
                w_ic = w[ic_mask]
                
                # 2. 초기 속도 손실 (Initial Displacement Loss)
                loss_ic_disp = torch.mean(u_ic**2 + v_ic**2 + w_ic**2)
                du_dt_ic = du_dt[ic_mask]
                dv_dt_ic = dv_dt[ic_mask]
                dw_dt_ic = dw_dt[ic_mask]

                # L_ic_velocity = (du/dt(t=0)-0)^2 + (dv/dt(t=0)-0)^2 + (dw/dt(t=0)-0)^2
                loss_ic_vel = torch.mean(du_dt_ic**2 + dv_dt_ic**2 + dw_dt_ic**2)
    
                ic_loss = loss_ic_disp + loss_ic_vel

        # 5. Boundary Condition (BC) Loss
        bc_loss = torch.tensor(0.0, **kwargs)
        if use_bc:
            final_xyz = xyzt[..., :3] + uvw_local
            final_y = final_xyz[..., self.up_index]
            
            # penetration_mask = final_y < self.ground_pos
            
            # penetration_loss = torch.tensor(0.0, **kwargs)
            # if torch.any(penetration_mask):
            #     # 관통한 깊이(ground_pos - final_y)의 제곱에 비례하는 손실 계산
            #     penetration_error = self.ground_pos - final_y[penetration_mask]
            #     penetration_loss = torch.mean(penetration_error**2)

            # 속도 제약-(Sticky Loss) PAC-NERF 에서 구현된거 그대로 사용 
            contact_epsilon = 1.0E-6 
            contact_mask = final_y < (self.ground_pos + contact_epsilon)
            
            sticky_loss = torch.tensor(0.0, **kwargs)
            if torch.any(contact_mask):

                velocity = torch.stack([du_dt, dv_dt, dw_dt], dim=-1)
                velocity_at_contact = velocity[contact_mask]
                
                # 속도의 제곱(squared norm)을 손실로 하여 0으로 수렴하도록 유도
                sticky_loss = torch.mean(torch.sum(velocity_at_contact**2, dim=-1))

            bc_loss = sticky_loss


        if use_vel and (displacement is not None) and (time is not None):
            uvw_unflat = uvw_global.reshape(time_dim, point_dim, -1)
            disp_unflat = displacement.reshape(time_dim, point_dim, -1)
            time_unflat = time.reshape(time_dim, point_dim, -1)
            delta_time = time_unflat[+1:] - time_unflat[:-1]
            vel_unflat = (uvw_unflat[+1:] - uvw_unflat[:-1]) / delta_time
            vel_gt_unflat = (disp_unflat[+1:] - disp_unflat[:-1]) / delta_time
            l1_loss = torch.mean((vel_unflat - vel_gt_unflat).abs())
            l2_loss = torch.mean((vel_unflat - vel_gt_unflat).square())
            vel_loss = l2_loss + l1_loss
        else:
            vel_loss = torch.tensor(0.0, **kwargs)

        # 7. Ground Truth (GT) Loss 
        gt_loss = torch.tensor(0.0, **kwargs)
        if displacement is not None:
            disp_flat = displacement.reshape(-1, 3)
            l1_loss = torch.mean((uvw_global - disp_flat).abs())
            l2_loss = torch.mean((uvw_global - disp_flat).square())
            gt_loss = l1_loss + l2_loss

        return {
            "pde_loss": pde_loss,
            "ic_loss":  ic_loss,
            "vel_loss": vel_loss,
            "gt_loss":  gt_loss,
            "bc_loss" : bc_loss,
        }


if __name__ == "__main__":
    model = NavierCauchy()
    xyz  = torch.randn(10, 512, 3)
    t    = torch.linspace(0, 1, 10).view(10, 1, 1).repeat(1, 512, 1)
    xyzt = torch.cat([xyz, t], dim = 2)

    loss = model.compute_loss(xyzt)
    print({k:f"{v.item():.3e}" for k,v in loss.items()})