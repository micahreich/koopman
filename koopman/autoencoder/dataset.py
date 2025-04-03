from typing import Union

import einops
import numpy as np
import torch
from torch.utils.data import Dataset


class KoopmanDataset(Dataset):
    def __init__(self,
                 x_hist: Union[np.ndarray, torch.Tensor],
                 u_hist: Union[np.ndarray, torch.Tensor],
                 ts: Union[np.ndarray, torch.Tensor],
                 pred_horizon: int,
                 dt: float,
                 train_split: float = 0.8):
        self.x_hist = torch.as_tensor(x_hist, dtype=torch.float32)
        self.u_hist = torch.as_tensor(u_hist, dtype=torch.float32)
        self.ts = torch.as_tensor(ts, dtype=torch.float32)

        assert len(self.x_hist.shape) == len(self.u_hist.shape) and len(self.x_hist.shape) in [2, 3], \
            "x_hist and uhist must have the same number of dimensions and be either 2 or 3 dimensional"

        assert pred_horizon > 0 and pred_horizon < self.x_hist.shape[1], \
            "pred_horizon must be greater than 0 and less than the length of the time series"

        if len(self.x_hist.shape) == 2:
            self.x_hist = self.x_hist.unsqueeze(0)
            self.u_hist = self.u_hist.unsqueeze(0)

        self.N_sys, self.H, self.nx = self.x_hist.shape
        _, _, self.nu = self.u_hist.shape

        self.pred_horizon = pred_horizon
        self.dt = dt

        self.train_split = train_split

        self.N_sys_train = int(self.N_sys * self.train_split)
        self.N_sys_eval = self.N_sys - self.N_sys_train

        self.indices_train = np.arange(self.N_sys_train)
        self.indices_eval = np.arange(self.N_sys_train, self.N_sys)

    def sample_eval_trajectories(self, n: int = 1):
        sys_idx = torch.randint(0, self.N_sys_eval, (n, ))
        sys_idx = self.indices_eval[sys_idx]

        return self.x_hist[sys_idx], self.u_hist[sys_idx], self.ts

    @staticmethod
    def flatten_batch(self, x_batch: torch.Tensor) -> torch.Tensor:
        assert x_batch.dim() == 3, "x_batch must be a 3D tensor"
        return einops.rearrange(x_batch, 'b h nx -> (b h) nx')

    def __len__(self):
        return self.N_sys_train * (self.H - self.pred_horizon)

    def __getitem__(self, idx):
        sys_idx = idx // (self.H - self.pred_horizon)
        sys_idx = self.indices_train[sys_idx]

        time_idx = idx % (self.H - self.pred_horizon)

        t_start = time_idx
        t_end = t_start + self.pred_horizon + 1

        x_vals = self.x_hist[sys_idx, t_start:t_end, :]
        u_vals = self.u_hist[sys_idx, t_start:t_end - 1, :]

        x0 = x_vals[0, :]
        u0 = u_vals[0, :]
        assert x0.shape == (self.nx, ) and u0.shape == (self.nu, )

        x_horizon = x_vals[1:, :]
        u_horizon = u_vals[1:, :]
        assert x_horizon.shape == (self.pred_horizon, self.nx) and u_horizon.shape == (self.pred_horizon - 1, self.nu)

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
