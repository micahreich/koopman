from typing import Callable, Optional, Tuple, Union

import numpy as np
from scipy.optimize import least_squares
from sklearn.linear_model import Ridge


class eDMD:
    def __init__(self):
        self.kpA = None
        self.kpB = None
        self.kpCs = None

    @staticmethod
    def apply_observable_to_history(xhist_3d: np.ndarray, obs_fn: Callable, *args, **kwargs) -> np.ndarray:
        xhist_flat = xhist_3d.reshape(-1, xhist_3d.shape[-1])
        zhist_flat = obs_fn(xhist_flat, *args, **kwargs)

        return zhist_flat.reshape(xhist_3d.shape[0], xhist_3d.shape[1], -1)

    def project_to_x(self, zhist: np.ndarray, nx) -> np.ndarray:
        if zhist.ndim == 1:
            return zhist[:nx]
        elif zhist.ndim == 2:
            return zhist[:, :nx]
        elif zhist.ndim == 3:
            return zhist[:, :, :nx]

    def predict_z_next(self, zs: np.ndarray, us: np.ndarray) -> np.ndarray:
        assert zs.ndim == us.ndim, "zs and us must have the same number of dimensions"
        is_batched = zs.ndim == 2
        if not is_batched:
            zs = zs[np.newaxis, :]
            us = us[np.newaxis, :]

        C_i_u_i_sum = np.einsum('b j, j m n -> b m n', us, self.kpCs)
        C_i_u_i_sum_x_zs = np.einsum('b m n, b n -> b m', C_i_u_i_sum, zs)

        out = zs @ self.kpA.T + us @ self.kpB.T + C_i_u_i_sum_x_zs

        if not is_batched:
            return np.squeeze(out, axis=0)

        return out

    def stable_ls(A, b, tol=1e-8):
        U, s, Vt = np.linalg.svd(A, full_matrices=False)
        # Invert s with thresholding: if s[i] is too small, set its inverse to zero
        s_inv = np.array([1 / si if si > tol else 0 for si in s])
        x = Vt.T @ np.diag(s_inv) @ U.T @ b
        return x

    def fit(self, zhist: np.ndarray, uhist: np.ndarray, nx, alpha=0.1):
        assert zhist.ndim == 3, "zhist must be a 3D array of shape (N, T, nz)"
        assert uhist.ndim == 3, "uhist must be a 3D array of shape (N, T-1, nu)"

        N, T, nz = zhist.shape
        _, _, nu = uhist.shape

        print(f"zhist shape: {zhist.shape}")
        print(f"uhist shape: {uhist.shape}")
        print(f"Number of samples: {N}")
        print(f"Number of time steps: {T}")
        print(f"Number of states: {nz}")
        print(f"Number of controls: {nu}")

        # Build data matrices
        B = N * (T - 1)

        zcurr = np.reshape(zhist[:, :-1, :], (B, -1))  # (B, nz)
        znext = np.reshape(zhist[:, 1:, :], (B, -1))  # (B, nz)
        ucurr = np.reshape(uhist, (B, nu))  # (B, nu)

        zcurr_expanded = zcurr[:, np.newaxis, :]  # (B, 1, nz)
        ucurr_expanded = ucurr[:, :, np.newaxis]  # (B, nu, 1)
        zucurr_mixed = np.reshape(ucurr_expanded * zcurr_expanded, (B, nz * nu))

        Y = np.concatenate([zcurr, ucurr, zucurr_mixed], axis=-1)
        Y_plus = znext

        W = np.linalg.solve(Y.T @ Y + 1e-4 * np.eye(Y.shape[1]), Y.T @ Y_plus)
        dynamics = W.T

        self.kpA = dynamics[:, :nz]  # (nz, nz)
        self.kpB = dynamics[:, nz:nz + nu]  # (nz, nu)
        kpC = dynamics[:, nz + nu:]  # (nz, nz * nu)
        self.kpCs = kpC.reshape(nz, nu, nz).transpose(1, 0, 2)  # (nu, nz, nz)

        print(f"kpA shape: {self.kpA.shape}")
        print(f"kpB shape: {self.kpB.shape}")
        print(f"Condition number of kpA: {np.linalg.cond(self.kpA)}")
        print(f"Condition number of kpB: {np.linalg.cond(self.kpB)}")

        return self.kpA, self.kpB, self.kpCs


def test1():
    # Try this with a simple linear system
    N = 50
    T = 6
    nx = 4
    nu = 2

    x0 = np.random.randn(N, nx)
    xs = np.empty((N, T, nx))
    xs[:, 0, :] = x0

    us = np.random.randn(N, T - 1, nu)

    A = np.random.randn(nx, nx)
    B = 10 * np.random.randn(nx, nu)
    Cs = np.random.randn(nu, nx, nx)

    for i in range(T - 1):
        C_i_u_i_sum = np.einsum('b j, j m n -> b m n', us[:, i, :], Cs)
        h = np.einsum('b m n, b n -> b m', C_i_u_i_sum, xs[:, i, :])
        xs[:, i + 1, :] = xs[:, i, :] @ A.T + us[:, i, :] @ B.T + h

    edmd = eDMD()
    kpA, kpB, kpCs = edmd.fit(xs, us, nx)
    print("kpA:", kpA)
    print("A:", A)

    print("kpB:", kpB)
    print("B:", B)

    print("kpCs:", kpCs)
    print("Cs:", Cs)


if __name__ == "__main__":
    test1()
