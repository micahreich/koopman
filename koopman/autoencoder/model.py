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
        use_layernorm=False,
        params_init: Optional[str] = None,
        params_init_args: dict = {},
    ):
        assert len(dims) >= 2

        super().__init__()
        self.use_layernorm = use_layernorm
        self.activation = activation()

        # Build layers
        self.layers = nn.ModuleList()
        self.layernorms = nn.ModuleList() if use_layernorm else None

        for i in range(len(dims) - 1):
            m = nn.Linear(dims[i], dims[i + 1])

            self.layers.append(m)

            if use_layernorm and i < len(dims) - 2:  # Don't apply LayerNorm on output layer
                self.layernorms.append(nn.LayerNorm(dims[i + 1]))

        if params_init is not None:
            weight_init[params_init](self.layers[-1].weight, **params_init_args)

    def forward(self, x):
        for i, layer in enumerate(self.layers[:-1]):
            x = layer(x)
            if self.use_layernorm:
                x = self.layernorms[i](x)
            x = self.activation(x)

        return self.layers[-1](x)


class KoopmanAutoencoder(nn.Module):
    def __init__(self,
                 nx: int,
                 nu: int,
                 nz: int,
                 pred_horizon: int,
                 params_init: str = 'eye',
                 params_init_args: dict = {},
                 horizon_loss_weight_x: float = 1.0,
                 horizon_loss_weight_z: float = 1.0,
                 L1_reg_weight: float = 1.0,
                 jacobian_reg_weight: float = 0.0,
                 hidden_dims: List[int] = [64],
                 activation=nn.Mish,
                 use_layernorm=False):
        super().__init__()

        assert nz > nx, "Latent space dimension must be greater than state space dimension"

        self.nx = nx  # Dimension of the state space
        self.nu = nu  # Dimension of the control space
        self.nz = nz  # Dimension of the latent space
        self.pred_horizon = pred_horizon  # Prediction horizon

        self.A = nn.Parameter(torch.empty((nz, nz)))
        weight_init[params_init](self.A, **params_init_args)

        self.B = nn.Parameter(torch.zeros((nz, nu)))
        # torch.nn.init.trunc_normal_(self.B, std=0.01, a=-1.0, b=1.0)
        # init.xavier_uniform_(self.B)

        self.Cs = nn.Parameter(torch.zeros((nu, nz, nz)))
        # torch.nn.init.trunc_normal_(self.Cs, std=0.01, a=-1.0, b=1.0)
        # init.xavier_uniform_(self.Cs)

        #     weight_init[params_init](self.Cs[i, :, :], **params_init_args)

        self.d = nn.Parameter(torch.zeros((nz, 1)))

        self.P = torch.zeros((nx, nz))
        self.P[:, :nx] = torch.eye(nx)

        self.gamma = 1.0
        self.horizon_loss_weights = self.gamma ** torch.arange(0, self.pred_horizon)

        # Weight different parts of the combined loss differently
        loss_weights_sum = horizon_loss_weight_x + horizon_loss_weight_z + L1_reg_weight + jacobian_reg_weight

        self.horizon_loss_weight_x = horizon_loss_weight_x / loss_weights_sum
        self.horizon_loss_weight_z = horizon_loss_weight_z / loss_weights_sum
        self.L1_reg_weight = L1_reg_weight / loss_weights_sum
        self.jacobian_reg_weight = jacobian_reg_weight / loss_weights_sum

        # Build the encoder (MLP); outputs the observables not including the original state,
        # which is concatenated on to observables during fwd pass
        dims = [nx] + hidden_dims + [nz - nx]

        self.encoder = MLP(dims, activation, use_layernorm)

        self.param_groups = {
            'observables': self.encoder.parameters(),
            'dynamics': [self.A, self.B, self.Cs, self.d],
        }

    def flatten_batch(self, x_batch: torch.Tensor) -> torch.Tensor:
        assert x_batch.dim() == 3, "x_batch must be a 3D tensor"
        return einops.rearrange(x_batch, 'b h nx -> (b h) nx')

    def unflatten_batch(self, batch_size, x_flat_batch: torch.Tensor) -> torch.Tensor:
        assert x_flat_batch.dim() == 2, "x_flat_batch must be a 2D tensor"
        return einops.rearrange(x_flat_batch, '(b h) nx -> b h nx', b=batch_size)

    def _dynamics_jacobian_norm(self, x0: torch.Tensor, u0: torch.Tensor, z0: torch.Tensor):
        device = x0.device
        B, _ = x0.shape
        x1_pred = self.project(self.predict_z_next(z0, u0))

        n_proj = 2
        vsx = torch.randn(B, n_proj, self.nx, requires_grad=True, device=device)

        grads_x = torch.zeros(B, device=device)
        grads_u = torch.zeros(B, device=device)

        # Estimate the squared Frobenius norm of the discrete-time dynamics Jacobians
        # using the Girard–Hutchinson randomized trace estimator (||J||_F^2 = Tr(JJ^T)),
        # see https://arxiv.org/abs/1908.02729

        for i in range(n_proj):
            vJx, vJu = torch.autograd.grad(x1_pred, (x0, u0), grad_outputs=vsx[:, i, :], create_graph=True)

            grads_x += 1 / n_proj * (torch.norm(vJx, 2, dim=-1) ** 2)
            grads_u += 1 / n_proj * (torch.norm(vJu, 2, dim=-1) ** 2)

        Jx_norm_mean = grads_x.mean(dim=0)
        Ju_norm_mean = grads_u.mean(dim=0)

        return Jx_norm_mean, Ju_norm_mean

    def loss(self, xs: Tuple[torch.Tensor, torch.Tensor], us: Tuple[torch.Tensor, torch.Tensor],
             zs: Tuple[torch.Tensor, torch.Tensor]):
        x0, x_horizon = xs
        u0, u_horizon = us
        z0, z_horizon = zs

        loss_pred = 0.0
        loss_bilinear = 0.0
        loss_l1 = 0.0

        # (2) Ensure that phi(x_{t+j+1}) = K z_{t+j} + B u_{t+j} + (sum_i C_i u_{t+j}) * z_{t+j} for j = 0, ..., H-1
        z_jm1 = z0
        u_jm1 = u0

        for j in range(0, self.pred_horizon):
            z_j_from_rollout = self.predict_z_next(z_jm1, u_jm1)
            z_j_gt = z_horizon[:, j, :]

            x_j_from_rollout = self.project(z_j_from_rollout)
            x_j_gt = x_horizon[:, j, :]

            loss_pred += F.mse_loss(x_j_from_rollout, x_j_gt, reduction='mean')
            loss_bilinear += F.mse_loss(z_j_from_rollout, z_j_gt, reduction='mean')

            z_jm1 = z_j_from_rollout

            if j < self.pred_horizon - 1:
                u_jm1 = u_horizon[:, j, :]

        loss_pred *= 1 / self.pred_horizon
        loss_bilinear *= 1 / self.pred_horizon

        # (2) Encourage sparsity in the matrix K to not have redundant information/reduce overfitting
        loss_l1 = torch.linalg.matrix_norm(self.A, ord=1)

        # # (3) Encourage the dynamics Jacobian to have small Frobenius norm
        # loss_jac_reg = torch.tensor(0.0, device=x0.device)
        # if self.jacobian_reg_weight > 0:
        #     Jx_norm_mean, Ju_norm_mean = self._dynamics_jacobian_norm(x0, u0, z0)
        #     loss_jac_reg = self.jacobian_reg_weight * (Jx_norm_mean + Ju_norm_mean)

        loss_dict = {
            'loss_pred': self.horizon_loss_weight_x * loss_pred,
            'loss_bilinear': self.horizon_loss_weight_z * loss_bilinear,
            'loss_l1': self.L1_reg_weight * loss_l1,
            # 'loss_jac': self.jacobian_reg_weight * loss_jac_reg,
        }

        return loss_dict['loss_pred'] + loss_dict['loss_bilinear'] + loss_dict['loss_l1'] + loss_dict[
            'loss_jac'], loss_dict

    def predict_z_next(self, z_jm1: torch.Tensor, u_jm1: torch.Tensor) -> torch.Tensor:
        # Use bilinear form of Koopman dynamcis:
        #   z_{t+1} = A z_t + B u_t + (sum_i C_i u_t) * z_t

        C_i_u_i_sum = torch.einsum('b j, j m n -> b m n', u_jm1, self.Cs)
        z_j_from_rollout = z_jm1 @ self.A.T + u_jm1 @ self.B.T + torch.einsum('b m n, b n -> b m', C_i_u_i_sum,
                                                                              z_jm1) + self.d.T

        return z_j_from_rollout

    def predict_x_next(self, x_jm1: torch.Tensor, u_jm1: torch.Tensor) -> torch.Tensor:
        z_jm1 = self.forward(x_jm1)
        z_j_from_rollout = self.predict_z_next(z_jm1, u_jm1)
        x_j_from_rollout = self.project(z_j_from_rollout)

        return x_j_from_rollout

    def project(self, z) -> torch.Tensor:
        if z.dim() == 2:
            return z[:, :self.nx]
        else:
            return z[:, :, :self.nx]

    def forward(self, x_batch):
        z_vals = self.encoder(x_batch)
        z_vals = torch.cat([x_batch, z_vals], dim=-1)

        return z_vals
