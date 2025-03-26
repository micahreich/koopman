from typing import Union

import numpy as np
import torch
from sklearn.decomposition import PCA
from torch.utils.data import Dataset


class KoopmanDataset(Dataset):
    def __init__(self, x_hist: Union[np.ndarray, torch.Tensor], u_hist: Union[np.ndarray, torch.Tensor],
                 ts: Union[np.ndarray, torch.Tensor], pred_horizon: int, dt: float):
        self.x_hist = torch.as_tensor(x_hist, dtype=torch.float32)
        self.u_hist = torch.as_tensor(u_hist, dtype=torch.float32)
        self.ts = torch.as_tensor(ts, dtype=torch.float32)

        assert len(self.x_hist.shape) == len(self.u_hist.shape) and len(self.x_hist.shape) in [2, 3], \
            "x_hist and uhist must have the same number of dimensions and be either 2 or 3 dimensional"

        if len(self.x_hist.shape) == 2:
            self.x_hist = self.x_hist.unsqueeze(0)
            self.u_hist = self.u_hist.unsqueeze(0)

        self.N_sys, self.H, self.nx = self.x_hist.shape
        _, _, self.nu = self.u_hist.shape

        self.pred_horizon = pred_horizon
        self.dt = dt

    def __len__(self):
        return self.N_sys * (self.H - self.pred_horizon)

    def __getitem__(self, idx):
        self.sys_idx = idx // (self.H - self.pred_horizon)
        self.time_idx = idx % (self.H - self.pred_horizon)

        t_start = self.time_idx
        t_end = t_start + self.pred_horizon + 1

        x_vals = self.x_hist[self.sys_idx, t_start:t_end, :]
        u_vals = self.u_hist[self.sys_idx, t_start:t_end - 1, :]

        assert x_vals.shape == (self.pred_horizon + 1, self.nx) and u_vals.shape == (self.pred_horizon, self.nu)

        return x_vals, u_vals


if __name__ == "__main__":
    # Test the KoopmanDataset to make sure it returns the correct values
    y = np.tile(np.arange(12, dtype=np.float32).reshape((-1, 1)), (1, 2))
    y = np.tile(y.reshape((1, 12, 2)), (3, 1, 1))
    y *= np.array([1., 10., 100.]).reshape((3, 1, 1))

    x_hist = y
    u_hist = y[:, :-1, [0]]

    dataset = KoopmanDataset(x_hist, u_hist, pred_horizon=10, dt=0.1)

    for i in range(len(dataset)):
        xvals, uvals = dataset[i]
        print("Sample", i)
        print(xvals.shape, uvals.shape)
        print("xvals")
        print(xvals)
        print("uvals")
        print(uvals)
        assert xvals.shape == (11, 2) and uvals.shape == (10, 1)
