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
        ground_pos: float = 0.0,
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
        return torch.exp(self.log_poissons)

    def property_parameters(self):
        yield self.log_density
        yield self.log_youngs
        yield self.log_poissons
    
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

            # The stress divergence term in Navier-Cauchy equation requires the second-order spatial derivatives.
            div_u = du_dx + dv_dy + dw_dz
            
            grad_div_u = autograd.grad(div_u, xyzt, torch.ones_like(div_u), create_graph=True)[0]
            d_div_u_dx = grad_div_u[..., 0]
            d_div_u_dy = grad_div_u[..., 1]
            d_div_u_dz = grad_div_u[..., 2]

            lap_u = autograd.grad(du_dx, xyzt, torch.ones_like(du_dx), create_graph=True)[0][...,  0] + \
                    autograd.grad(du_dy, xyzt, torch.ones_like(du_dy), create_graph=True)[0][..., 1] + \
                    autograd.grad(du_dz, xyzt, torch.ones_like(du_dz), create_graph=True)[0][..., 2]

            lap_v = autograd.grad(dv_dx, xyzt, torch.ones_like(dv_dx), create_graph=True)[0][..., 0] + \
                    autograd.grad(dv_dy, xyzt, torch.ones_like(dv_dy), create_graph=True)[0][..., 1] + \
                    autograd.grad(dv_dz, xyzt, torch.ones_like(dv_dz), create_graph=True)[0][..., 2]

            lap_w = autograd.grad(dw_dx, xyzt, torch.ones_like(dw_dx), create_graph=True)[0][..., 0] + \
                    autograd.grad(dw_dy, xyzt, torch.ones_like(dw_dy), create_graph=True)[0][..., 1] + \
                    autograd.grad(dw_dz, xyzt, torch.ones_like(dw_dz), create_graph=True)[0][..., 2]

            # The Navier-Cauchy equation residuals
            pde_x = self.density * du_dt2 - ( (lmbda + mu) * d_div_u_dx + mu * lap_u )
            pde_y = self.density * dv_dt2 - ( (lmbda + mu) * d_div_u_dy + mu * lap_v )
            pde_z = self.density * dw_dt2 - ( (lmbda + mu) * d_div_u_dz + mu * lap_w )

            # The gravity term.
            pde_y -= self.density * self.gravity
            
            if f_ext is not None:
                pde_x -= f_ext[:, 0]
                pde_y -= f_ext[:, 1]
                pde_z -= f_ext[:, 2]

            pde_loss = torch.mean(pde_x**2 + pde_y**2 + pde_z**2)

        # 4. Initial Condition (IC) Loss
        ic_mask = xyzt[..., 3] == 0.0
        if use_ic and (ic_mask.sum() > 0):
            if xyzt.ndim == 2:
                ic_u = self.forward(xyzt[ic_mask]).local_branch.reshape(-1, 3)
            elif xyzt.ndim == 3:
                ic_u = uvw_local.reshape(time_dim, point_dim, -1)[ic_mask]
            u0 = torch.zeros_like(ic_u)
            ic_loss = torch.nn.functional.mse_loss(ic_u, u0)
        else:
            ic_loss = torch.tensor(0.0, **kwargs)

        # 5. Boundary Condition (BC) Loss
        bc_loss = torch.tensor(0.0, **kwargs)
        if use_bc:
            # 5.1. y-coordinate of the current position
            # Initial y + y-direction displacement
            y_initial = xyzt[..., self.up_index].reshape(-1, 1)
            y_current = y_initial + v

            # 5.2. Penetration Loss
            penetration = torch.relu(self.ground_pos - y_current)
            loss_penetration = torch.mean(penetration**2) 

            # 5.3. Calculate the orthogonal stress in the y-direction.
            lmbda = (self.youngs * self.poissons) / ((1 + self.poissons) * (1 - 2 * self.poissons))
            mu = self.youngs / (2 * (1 + self.poissons))
            div_u = du_dx + dv_dy + dw_dz  # strain: How much the surrounding space has moved due to the displacement.
            epsilon_yy = dv_dy
            sigma_y = lmbda * div_u + 2 * mu * epsilon_yy

            # 5.4. Complementarity and stress conditions:
            # C1: When `y_current ≈ ground_pos`, the stress must be compression.
            contact_mask = (y_current <= self.ground_pos).float() 
            loss_tensile_stress = torch.mean((contact_mask * torch.relu(sigma_y))**2) # 거의 발동 안됨

            # C2: When `y_current > ground_pos`, the stress must be zero.
            no_contact_mask = (y_current > self.ground_pos).float()
            loss_no_contact_stress = torch.mean((no_contact_mask * sigma_y)**2)
            
            # The final BC loss is the sum of the terms.
            bc_loss = loss_penetration # + loss_no_contact_stress
            
        # 6. Velocity-Driven GT Loss
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
