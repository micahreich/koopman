from dataclasses import dataclass, field
from typing import Union

import einops
import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass
class KoopmanDatasetStats:
    x_mean: torch.Tensor = field(init=False)
    x_std: torch.Tensor = field(init=False)
    u_mean: torch.Tensor = field(init=False)
    x_std: torch.Tensor = field(init=False)
    nx: int = field(init=False)
    nu: int = field(init=False)
    dt: float = field(init=False)
    N_sys: int = field(init=False)
    H: int = field(init=False)

    @staticmethod
    def from_tensors(x_hist: torch.Tensor, u_hist: torch.Tensor, dt: float) -> 'KoopmanDatasetStats':
        stats = KoopmanDatasetStats()
        stats.x_mean = (x_hist.mean(dim=(0, 1))).float()
        stats.x_std = (x_hist.std(dim=(0, 1)) + 1e-8).float()  # Avoid division by zero
        stats.u_mean = (u_hist.mean(dim=(0, 1))).float()
        stats.u_std = (u_hist.std(dim=(0, 1)) + 1e-8).float()  # Avoid division by zero

        stats.nx = x_hist.shape[-1]
        stats.nu = u_hist.shape[-1]
        stats.dt = dt

        N_sys, H, _ = x_hist.shape
        stats.N_sys = N_sys
        stats.H = H

        return stats

    def __str__(self):
        return f"""KoopmanDatasetStats(
nx={self.nx},
nu={self.nu},
dt={self.dt},
x_mean={self.x_mean},
x_std={self.x_std},
u_mean={self.u_mean},
u_std={self.u_std}),
N_sys={self.N_sys},
H={self.H}"""

    def _reshape_stats(self, s: torch.Tensor, d: int, device) -> torch.Tensor:
        return s.view(*([1] * (d - 1)), -1).to(device)

    def normalize_x(self, x: torch.Tensor) -> torch.Tensor:
        mean = self._reshape_stats(self.x_mean, x.dim(), x.device)
        std = self._reshape_stats(self.x_std, x.dim(), x.device)
        return (x - mean) / std

    def normalize_u(self, u: torch.Tensor) -> torch.Tensor:
        mean = self._reshape_stats(self.u_mean, u.dim(), u.device)
        std = self._reshape_stats(self.u_std, u.dim(), u.device)
        return (u - mean) / std

    def denormalize_x(self, x: torch.Tensor) -> torch.Tensor:
        mean = self._reshape_stats(self.x_mean, x.dim(), x.device)
        std = self._reshape_stats(self.x_std, x.dim(), x.device)
        return x * std + mean

    def denormalize_u(self, u: torch.Tensor) -> torch.Tensor:
        mean = self._reshape_stats(self.u_mean, u.dim(), u.device)
        std = self._reshape_stats(self.u_std, u.dim(), u.device)
        return u * std + mean


class KoopmanDataset(Dataset):
    def __init__(self,
                 x_hist: torch.Tensor,
                 u_hist: torch.Tensor,
                 ts: torch.Tensor,
                 stats: KoopmanDatasetStats,
                 pred_horizon: int,
                 normalize: bool,
                 train_split: float = 0.8):
        # Make sure data is f32
        self.x_hist = x_hist.float()
        self.u_hist = u_hist.float()
        self.ts = ts.float()

        assert len(self.x_hist.shape) == len(self.u_hist.shape) and len(self.x_hist.shape) in [2, 3], \
            "x_hist and uhist must have the same number of dimensions and be either 2 or 3 dimensional"

        assert pred_horizon > 0 and pred_horizon < self.x_hist.shape[1], \
            "pred_horizon must be greater than 0 and less than the length of the time series"

        if len(self.x_hist.shape) == 2:
            self.x_hist = self.x_hist.unsqueeze(0)
            self.u_hist = self.u_hist.unsqueeze(0)

        # self.N_sys, self.H, self.stats.nx = self.x_hist.shape
        # _, _, self.stats.nu = self.u_hist.shape

        self.pred_horizon = pred_horizon
        self.train_split = train_split
        self.stats = stats

        self.normalize = normalize
        if self.normalize:
            self.x_hist = self.stats.normalize_x(self.x_hist)
            self.u_hist = self.stats.normalize_u(self.u_hist)

        self.N_sys_train = int(self.stats.N_sys * self.train_split)
        self.N_sys_eval = self.stats.N_sys - self.N_sys_train

        self.indices_train = np.arange(self.N_sys_train)
        self.indices_eval = np.arange(self.N_sys_train, self.stats.N_sys)

    def sample_eval_trajectories(self, n: int = 1):
        sys_idx = torch.randint(0, self.N_sys_eval, (n, ))
        sys_idx = self.indices_eval[sys_idx]

        return self.x_hist[sys_idx], self.u_hist[sys_idx], self.ts

    @staticmethod
    def flatten_batch(self, x_batch: torch.Tensor) -> torch.Tensor:
        assert x_batch.dim() == 3, "x_batch must be a 3D tensor"
        return einops.rearrange(x_batch, 'b h nx -> (b h) nx')

    def __len__(self):
        return self.N_sys_train * (self.stats.H - self.pred_horizon)

    def __getitem__(self, idx):
        sys_idx = idx // (self.stats.H - self.pred_horizon)
        sys_idx = self.indices_train[sys_idx]

        time_idx = idx % (self.stats.H - self.pred_horizon)

        t_start = time_idx
        t_end = t_start + self.pred_horizon + 1

        x_vals = self.x_hist[sys_idx, t_start:t_end, :]
        u_vals = self.u_hist[sys_idx, t_start:t_end - 1, :]

        x0 = x_vals[0, :]
        u0 = u_vals[0, :]
        assert x0.shape == (self.stats.nx, ) and u0.shape == (self.stats.nu, )

        x_horizon = x_vals[1:, :]
        u_horizon = u_vals[1:, :]
        assert x_horizon.shape == (self.pred_horizon, self.stats.nx) and u_horizon.shape == (self.pred_horizon - 1,
                                                                                             self.stats.nu)

        return x0, u0, x_horizon, u_horizon


if __name__ == "__main__":
    # Test the KoopmanDataset to make sure it returns the correct values
    ts = np.arange(0, 12, 0.1, dtype=np.float32)

    y = np.tile(np.arange(12, dtype=np.float32).reshape((-1, 1)), (1, 2))
    y = np.tile(y.reshape((1, 12, 2)), (3, 1, 1))
    y *= np.array([1., 10., 100.]).reshape((3, 1, 1))

    x_hist = y
    u_hist = y[:, :-1, [0]]

    pred_horizon = 3
    dataset = KoopmanDataset(x_hist, u_hist, ts, pred_horizon=pred_horizon, dt=0.1)

    for i in range(len(dataset)):
        xvals, uvals = dataset[i]
        print("Sample", i)
        print(xvals.shape, uvals.shape)
        print("xvals")
        print(xvals)
        print("uvals")
        print(uvals)
        assert xvals.shape == (pred_horizon + 1, 2) and uvals.shape == (pred_horizon, 1)

    xvals, uvals, _ = dataset.sample_eval_trajectories(1)
    print("Sample eval trajectories")
    print(xvals.shape, uvals.shape)
    print("xvals")
    print(xvals)
    print("uvals")
    print(uvals)
