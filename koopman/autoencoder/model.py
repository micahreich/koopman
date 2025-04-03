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
                 H: int,
                 params_init: str = 'xavier_uniform',
                 params_init_args: dict = {},
                 projection_loss_weight: float = 1.0,
                 horizon_loss_weight: float = 1.0,
                 L1_loss_weight: float = 1.0,
                 hidden_dims: List[int] = [64],
                 activation=nn.Mish,
                 use_layernorm=False):
        super().__init__()

        assert nz > nx, "Latent space dimension must be greater than state space dimension"

        self.nx = nx  # Dimension of the state space
        self.nu = nu  # Dimension of the control space
        self.nz = nz  # Dimension of the latent space
        self.H = H  # Prediction horizon

        self.A = nn.Parameter(torch.empty((nz, nz)))
        weight_init[params_init](self.A, **params_init_args)

        self.B = nn.Parameter(torch.empty((nz, nu)))
        weight_init[params_init](self.B, **params_init_args)

        self.Cs = nn.Parameter(torch.empty((nu, nz, nz)))
        for i in range(nu):
            weight_init[params_init](self.Cs[i, :, :], **params_init_args)

        self.P = torch.zeros((nx, nz))
        self.P[:, :nx] = torch.eye(nx)

        self.gamma = 0.9  #1.0
        self.horizon_loss_weights = self.gamma ** torch.arange(0, self.H)

        # Weight different parts of the combined loss differently
        loss_weights_sum = projection_loss_weight + horizon_loss_weight + L1_loss_weight

        self.projection_loss_weight = projection_loss_weight / loss_weights_sum
        self.horizon_loss_weight = horizon_loss_weight / loss_weights_sum
        self.L1_loss_weight = L1_loss_weight / loss_weights_sum

        # Build the encoder (MLP); outputs the observables not including the original state,
        # which is concatenated on to observables during fwd pass
        dims = [nx] + hidden_dims + [nz - nx]

        self.encoder = MLP(dims, activation, use_layernorm)

        self.param_groups = {
            'observables': self.encoder.parameters(),
            'dynamics': [self.A, self.B, self.Cs],
        }

    @staticmethod
    def flatten_batch(x_batch: torch.Tensor) -> torch.Tensor:
        assert x_batch.dim() == 3, "x_batch must be a 3D tensor"
        return einops.rearrange(x_batch, 'b h nx -> (b h) nx')

    def unflatten_batch(self, batch_size, x_flat_batch: torch.Tensor) -> torch.Tensor:
        assert x_flat_batch.dim() == 2, "x_flat_batch must be a 2D tensor"
        return einops.rearrange(x_flat_batch, '(b h) nx -> b h nx', b=batch_size)

    def _dynamics_jacobian_norm(self, x0: torch.Tensor, u0: torch.Tensor, z0: torch.Tensor) -> torch.Tensor:
        B, _ = x0.shape
        x1_pred = self.project(self.predict_z_next(z0, u0))

        n_proj = 2
        vsx = torch.randn(B, n_proj, self.nx, requires_grad=True)

        grads_x = torch.zeros(B, )
        grads_u = torch.zeros(B, )

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
        loss_l1 = 0.0

        # (2) Ensure that phi(x_{t+j+1}) = K z_{t+j} + B u_{t+j} + (sum_i C_i u_{t+j}) * z_{t+j} for j = 0, ..., H-1
        z_jm1 = z0
        u_jm1 = u0

        for j in range(0, self.H):
            z_j_from_rollout = self.predict_z_next(z_jm1, u_jm1)
            z_j_from_encoding = z_horizon[:, j, :]

            loss_pred += self.horizon_loss_weights[j] * F.mse_loss(
                z_j_from_encoding, z_j_from_rollout, reduction='mean')

            z_jm1 = z_j_from_rollout

            if j < self.H - 1:
                u_jm1 = u_horizon[:, j, :]

        # (2) Encourage sparsity in the matrix K to not have redundant information/reduce overfitting
        loss_l1 = torch.linalg.matrix_norm(self.A, ord=1)

        loss_by_parts = {
            'loss_pred': self.horizon_loss_weight * loss_pred / self.H,
            'loss_l1': self.L1_loss_weight * loss_l1,
        }

        return loss_by_parts['loss_pred'] + loss_by_parts['loss_l1'], loss_by_parts

    def predict_z_next(self, z_jm1: torch.Tensor, u_jm1: torch.Tensor) -> torch.Tensor:
        C_i_u_i_sum = torch.einsum('b j, j m n -> b m n', u_jm1, self.Cs)
        z_j_from_rollout = z_jm1 @ self.A.T + u_jm1 @ self.B.T + torch.einsum('b m n, b n -> b m', C_i_u_i_sum, z_jm1)

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
