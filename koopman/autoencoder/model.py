from typing import List, Optional, Tuple, Union

import einops
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.utils.data import Dataset

weight_init = {
    'xavier_uniform': init.xavier_uniform_,
    'xavier_normal': init.xavier_normal_,
    'kaiming_uniform': init.kaiming_uniform_,
    'kaiming_normal': init.kaiming_normal_,
    'orthogonal': init.orthogonal_,
    'uniform': init.uniform_,
    'normal': init.normal_,
    'constant': init.constant_,
    'zeros': init.zeros_,
    'eye': init.eye_,
    'dirac': init.dirac_,
    'xavier_uniform_': init.xavier_uniform_,
    'xavier_normal_': init.xavier_normal_,
    'kaiming_uniform_': init.kaiming_uniform_,
    'kaiming_normal_': init.kaiming_normal_,
}


class MLP(nn.Module):
    def __init__(
        self,
        dims: List[int],
        activation=nn.Mish,
    ):
        assert len(dims) >= 2

        super().__init__()
        self.activation = activation()

        # Build layers
        self.layers = nn.ModuleList()
        for i in range(len(dims) - 1):
            m = nn.Linear(dims[i], dims[i + 1])
            self.layers.append(m)

    def forward(self, x):
        for i, layer in enumerate(self.layers[:-1]):
            x = layer(x)
            x = self.activation(x)

        return self.layers[-1](x)


class KoopmanAutoencoder(nn.Module):
    def __init__(self,
                 nx: int,
                 nu: int,
                 nz: int,
                 dt: float,
                 horizon_loss_weight: float = 1.0,
                 energy_loss_weight: float = 1.0,
                 hidden_dims: List[int] = [64],
                 activation=nn.Mish):
        super().__init__()

        assert nz > nx, "Latent space dimension must be greater than state space dimension"

        self.nx = nx  # Dimension of the state space
        self.nu = nu  # Dimension of the control space
        self.nz = nz  # Dimension of the latent space
        self.dt = dt  # Time step

        # A matrix is stable by construction
        self.Q = nn.Parameter(torch.empty((nz, nz)))
        self.C = nn.Parameter(-torch.ones((nz, )))
        torch.nn.init.eye_(self.Q)
        # self.A = nn.Parameter(torch.zeros((nz, nz)))
        # torch.nn.init.eye_(self.A)

        self.B = nn.Parameter(torch.zeros((nz, nu)))
        # torch.nn.init.xavier_normal_(self.B)

        self.Cs = nn.Parameter(torch.zeros((nz, nu * nz)))
        # torch.nn.init.xavier_normal_(self.Cs)

        # self.d = nn.Parameter(torch.zeros((nz, 1)))

        self.P = torch.zeros((nx, nz))
        self.P[:, :nx] = torch.eye(nx)

        self.horizon_loss_weight = horizon_loss_weight
        self.energy_loss_weight = energy_loss_weight

        # Build the encoder (MLP); outputs the observables not including the original state,
        # which is concatenated on to observables during fwd pass
        dims = [nx] + hidden_dims + [nz - nx]

        self.encoder = MLP(dims, activation)

    def flatten_batch(self, x_batch: torch.Tensor) -> torch.Tensor:
        assert x_batch.dim() == 3, "x_batch must be a 3D tensor"
        return einops.rearrange(x_batch, 'b h nx -> (b h) nx')

    def unflatten_batch(self, batch_size, x_flat_batch: torch.Tensor) -> torch.Tensor:
        assert x_flat_batch.dim() == 2, "x_flat_batch must be a 2D tensor"
        return einops.rearrange(x_flat_batch, '(b h) nx -> b h nx', b=batch_size)

    def get_dynamics(self):
        A_cont = self.Q - self.Q.T + torch.diag(-torch.exp(self.C))
        A = torch.matrix_exp(A_cont * self.dt)

        return A, self.B, self.Cs

    # def _dynamics_jacobian_norm(self, x0: torch.Tensor, u0: torch.Tensor, z0: torch.Tensor):
    #     device = x0.device
    #     B, _ = x0.shape
    #     x1_pred = self.project(self.predict_z_next(z0, u0))

    #     n_proj = 2
    #     vsx = torch.randn(B, n_proj, self.nx, requires_grad=True, device=device)

    #     grads_x = torch.zeros(B, device=device)
    #     grads_u = torch.zeros(B, device=device)

    #     # Estimate the squared Frobenius norm of the discrete-time dynamics Jacobians
    #     # using the Girard–Hutchinson randomized trace estimator (||J||_F^2 = Tr(JJ^T)),
    #     # see https://arxiv.org/abs/1908.02729

    #     for i in range(n_proj):
    #         vJx, vJu = torch.autograd.grad(x1_pred, (x0, u0), grad_outputs=vsx[:, i, :], create_graph=True)

    #         grads_x += 1 / n_proj * (torch.norm(vJx, 2, dim=-1) ** 2)
    #         grads_u += 1 / n_proj * (torch.norm(vJu, 2, dim=-1) ** 2)

    #     Jx_norm_mean = grads_x.mean(dim=0)
    #     Ju_norm_mean = grads_u.mean(dim=0)

    #     return Jx_norm_mean, Ju_norm_mean

    # def get_A(self):
    #     A_cont = self.Q - self.Q.T + torch.diag(-torch.exp(self.C))
    #     A = torch.matrix_exp(A_cont * self.dt)
    #     return A

    def loss(self, xs: Tuple[torch.Tensor, torch.Tensor], us: Tuple[torch.Tensor, torch.Tensor],
             zs: Tuple[torch.Tensor, torch.Tensor], pred_horizon):
        x0, x_horizon = xs
        u0, u_horizon = us
        z0, z_horizon = zs

        loss_pred = 0.0
        loss_energy = 0.0

        # (2) Ensure that phi(x_{t+j+1}) = K z_{t+j} + B u_{t+j} + (sum_i C_i u_{t+j}) * z_{t+j} for j = 0, ..., H-1
        z_jm1 = z0
        u_jm1 = u0

        theta_jm1, omega_jm1 = z0[:, 0, None], z0[:, 1, None]

        E_jm1 = 1 / 2 * torch.square(omega_jm1) + 9.81 * (1. - torch.cos(theta_jm1))

        # A = self.get_A()
        A, B, Cs = self.get_dynamics()

        for j in range(0, pred_horizon):
            z_j_from_rollout = self.predict_z_next(z_jm1, u_jm1, A, B, Cs)
            # z_j_gt = z_horizon[:, j, :]

            x_j_from_rollout = self.project(z_j_from_rollout)
            x_j_gt = x_horizon[:, j, :]

            theta_j, omega_j = x_j_from_rollout[:, 0, None], x_j_from_rollout[:, 1, None]
            Ej = 1 / 2 * torch.square(omega_j) + 9.81 * (1. - torch.cos(theta_j))

            delta_E = Ej - E_jm1
            delta_E_gt = (u_jm1 * omega_jm1) * self.dt

            loss_pred += F.mse_loss(x_j_from_rollout, x_j_gt, reduction='mean')
            loss_energy += F.mse_loss(delta_E, delta_E_gt, reduction='mean')

            z_jm1 = z_j_from_rollout
            E_jm1 = Ej
            theta_jm1, omega_jm1 = theta_j, omega_j

            if j < pred_horizon - 1:
                u_jm1 = u_horizon[:, j, :]

        loss_pred *= 1 / pred_horizon
        loss_energy *= 1 / pred_horizon

        # (2) Encourage sparsity in the matrix K to not have redundant information/reduce overfitting
        # loss_l1 = torch.linalg.matrix_norm(self.A, ord=1)
        # loss_l1 = torch.tensor(0.0, device=x0.device)

        # # (3) Encourage the dynamics Jacobian to have small Frobenius norm
        # loss_jac_reg = torch.tensor(0.0, device=x0.device)
        # if self.jacobian_reg_weight > 0:
        #     Jx_norm_mean, Ju_norm_mean = self._dynamics_jacobian_norm(x0, u0, z0)
        #     loss_jac_reg = self.jacobian_reg_weight * (Jx_norm_mean + Ju_norm_mean)

        loss_dict = {
            'loss_pred': self.horizon_loss_weight * loss_pred,
            'loss_energy': self.energy_loss_weight * loss_energy,
            # 'loss_bilinear': self.horizon_loss_weight_z * loss_bilinear,
            # 'loss_l1': self.L1_reg_weight * loss_l1,
            # 'loss_jac': self.jacobian_reg_weight * loss_jac_reg,
        }

        return loss_dict['loss_pred'] + loss_dict['loss_energy'], loss_dict

    def predict_z_next(self, z_jm1: torch.Tensor, u_jm1: torch.Tensor, A, B, Cs) -> torch.Tensor:
        # Use bilinear form of Koopman dynamcis:
        #   z_{t+1} = A z_t + B u_t + (sum_i C_i u_t) * z_t
        is_batched = z_jm1.dim() == 2 and u_jm1.dim() == 2
        if not is_batched:
            z_jm1 = z_jm1.unsqueeze(0)
            u_jm1 = u_jm1.unsqueeze(0)

        u_z_kron = torch.einsum('bi,bj->bij', u_jm1, z_jm1).reshape(-1, self.nu * self.nz)
        z_j_from_rollout = z_jm1 @ A.T + u_jm1 @ B.T + u_z_kron @ Cs.T

        if not is_batched:
            return z_j_from_rollout.squeeze(0)

        return z_j_from_rollout

    # def predict_z_next(self, z_jm1: torch.Tensor, u_jm1: torch.Tensor) -> torch.Tensor:
    #     # Use bilinear form of Koopman dynamcis:
    #     #   z_{t+1} = A z_t + B u_t + (sum_i C_i u_t) * z_t

    #     is_batched = z_jm1.dim() == 2 and u_jm1.dim() == 2
    #     if not is_batched:
    #         z_jm1 = z_jm1.unsqueeze(0)
    #         u_jm1 = u_jm1.unsqueeze(0)

    #     C_i_u_i_sum = torch.einsum('b j, j m n -> b m n', u_jm1, self.Cs)
    #     z_j_from_rollout = z_jm1 @ self.A.T + u_jm1 @ self.B.T + torch.einsum('b m n, b n -> b m', C_i_u_i_sum,
    #                                                                           z_jm1) + self.d.T

    #     if not is_batched:
    #         return z_j_from_rollout.squeeze(0)

    #     return z_j_from_rollout

    def predict_x_next(self, x_jm1: torch.Tensor, u_jm1: torch.Tensor) -> torch.Tensor:
        z_jm1 = self.forward(x_jm1)
        z_j_from_rollout = self.predict_z_next(z_jm1, u_jm1)
        x_j_from_rollout = self.project(z_j_from_rollout)

        return x_j_from_rollout

    def project(self, z) -> torch.Tensor:
        if z.dim() == 1:
            return z[:self.nx]
        elif z.dim() == 2:
            return z[:, :self.nx]
        else:
            return z[:, :, :self.nx]

    def forward(self, x_batch):
        z_vals = self.encoder(x_batch)
        z_vals = torch.cat([x_batch, z_vals], dim=-1)

        return z_vals
