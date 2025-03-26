from typing import List, Optional, Union

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
        params_init: str = 'xavier_uniform',
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
            # weight_init[params_init](m.weight, **params_init_args)

            self.layers.append(m)

            if use_layernorm and i < len(dims) - 2:  # Don't apply LayerNorm on output layer
                self.layernorms.append(nn.LayerNorm(dims[i + 1]))

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

        # Build the encoder (MLP)
        dims = [nx] + hidden_dims + [nz]

        self.encoder = MLP(dims, activation, use_layernorm, params_init='orthogonal')

    def loss(self, x_batch, u_batch, z_batch):
        B, _, _ = x_batch.shape

        # z_vals is the latent representation of all x_vals, i.e.
        #   z_vals[b, 1, :] = phi(x_vals[b, 1, :])
        #   z_vals[b, 2, :] = phi(x_vals[b, 2, :])
        #   ...
        #   z_vals[b, H, :] = phi(x_vals[b, H, :])

        assert u_batch.shape == (B, self.H, self.nu)
        assert x_batch.shape == (B, self.H + 1, self.nx)
        assert z_batch.shape == (B, self.H + 1, self.nz)

        loss = 0.0

        # (1) Ensure that the state is embedded in the latent vector
        x_vals_from_projection = self.project(z_batch)
        loss += self.projection_loss_weight * F.mse_loss(x_batch, x_vals_from_projection, reduction='mean')

        # (2) Ensure that phi(x_{t+j+1}) = K z_{t+j} + B u_{t+j} + (sum_i C_i u_{t+j}) * z_{t+j} for j = 0, ..., H-1
        z_jm1 = z_batch[:, 0, :]

        for j in range(1, self.H + 1):
            u_jm1 = u_batch[:, j - 1, :]

            z_j_from_rollout = self.predict_z_next(z_jm1, u_jm1)
            z_j_from_encoding = z_batch[:, j, :]

            loss += self.horizon_loss_weight * (self.horizon_loss_weights[j - 1] *
                                                F.mse_loss(z_j_from_encoding, z_j_from_rollout, reduction='mean'))
            z_jm1 = z_j_from_rollout

        # (3) Encourage sparsity in the matrix K to not have redundant information/reduce overfitting
        loss += self.L1_loss_weight * torch.linalg.matrix_norm(self.A, ord=1)

        return loss

    def predict_z_next(self, z_jm1: torch.Tensor, u_jm1: torch.Tensor) -> torch.Tensor:
        C_i_u_i_sum = torch.einsum('b j, j m n -> b m n', u_jm1, self.Cs)
        z_j_from_rollout = z_jm1 @ self.A.T + u_jm1 @ self.B.T + torch.einsum('b m n, b n -> b m', C_i_u_i_sum, z_jm1)

        return z_j_from_rollout

    def project(self, z) -> torch.Tensor:
        if z.dim() == 2:
            return z[:, :self.nx]
        else:
            return z[:, :, :self.nx]

    def forward(self, x_batch):
        # B, _, _ = x_batch.shape

        # x_batch = einops.rearrange(x_batch, 'b h nx -> (b h) nx')

        z_vals = self.encoder(x_batch)
        # z_vals = einops.rearrange(z_vals, '(b h) nz -> b h nz', b=B)

        return z_vals
