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

        _, nu = us.shape
        _, nz = zs.shape

        u_z_kron = np.einsum('bi,bj->bij', us, zs).reshape(-1, nu * nz)

        out = zs @ self.kpA.T + us @ self.kpB.T

        if self.kpCs is not None:
            out += u_z_kron @ self.kpCs.T

        if not is_batched:
            return np.squeeze(out, axis=0)

        return out

    def fit(self, zhist: np.ndarray, uhist: np.ndarray, bilinear=True):
        assert zhist.ndim == 3, "zhist must be a 3D array of shape (N, T, nz)"
        assert uhist.ndim == 3, "uhist must be a 3D array of shape (N, T-1, nu)"

        N, T, nz = zhist.shape
        _, _, nu = uhist.shape

        print(f"Fitting EDMD model with data:")
        print(f"\t{zhist.shape=}")
        print(f"\t{uhist.shape=}")
        print(f"\t# of trajectories: {N}")
        print(f"\t# of time steps: {T}")
        print(f"\t# of lifted states: {nz}")
        print(f"\t# of controls: {nu}")

        # Build data matrices
        B = N * (T - 1)

        z_curr = np.reshape(zhist[:, :-1, :], (B, -1))  # (B, nz)
        z_next = np.reshape(zhist[:, 1:, :], (B, -1))  # (B, nz)
        u_curr = np.reshape(uhist, (B, nu))  # (B, nu)

        if bilinear:
            u_z_kron = np.einsum('bi,bj->bij', u_curr, z_curr).reshape(-1, nu * nz)  # (B, nu * nz)
            Y = np.concatenate([z_curr, u_curr, u_z_kron], axis=-1)
        else:
            Y = np.concatenate([z_curr, u_curr], axis=-1)

        Y_plus = z_next

        # Want to solve min ||Y' - Y @ W||_F^2
        W = np.linalg.pinv(Y) @ Y_plus
        dynamics = W.T

        self.kpA = dynamics[:, :nz]  # (nz, nz)
        self.kpB = dynamics[:, nz:nz + nu]  # (nz, nu)

        # Check condition number
        print(f"Fitting results:")
        print(f"\t{self.kpA.shape=}")
        print(f"\t{self.kpB.shape=}")
        print(f"\tA condition number: {np.linalg.cond(self.kpA)}")
        print(f"\tB condition number: {np.linalg.cond(self.kpB)}")

        if bilinear:
            self.kpCs = dynamics[:, nz + nu:]  # (nz, nz * nu)
            return self.kpA, self.kpB, self.kpCs
        else:
            self.kpCs = None
            return self.kpA, self.kpB


def test1():
    N = 50
    T = 6
    nx = 10
    nu = 2

    x0 = np.random.randn(N, nx)
    xs = np.empty((N, T, nx))
    xs[:, 0, :] = x0

    us = np.random.randn(N, T - 1, nu)

    A = np.random.randn(nx, nx)
    B = 10 * np.random.randn(nx, nu)
    Cs = np.random.randn(nx, nu * nx)

    for i in range(T - 1):
        u_x_kron = np.einsum('b j, b k -> b j k', us[:, i, :], xs[:, i, :]).reshape(-1, nu * nx)
        xs[:, i + 1, :] = xs[:, i, :] @ A.T + us[:, i, :] @ B.T + u_x_kron @ Cs.T

    edmd = eDMD()
    kpA, kpB, kpCs = edmd.fit(xs, us, nx)

    assert np.allclose(kpA, A), "kpA does not match A"
    assert np.allclose(kpB, B), "kpB does not match B"
    assert np.allclose(kpCs, Cs), "kpCs does not match Cs"

    i = np.random.randint(0, N)
    xhist = xs[i, :, :]
    uhist = us[i, :, :]

    xnext_true = xhist[1:, :]
    xnext_pred = edmd.predict_z_next(xhist[:-1, :], uhist)

    assert np.allclose(xnext_true, xnext_pred), "Predicted next state does not match true next state"

    print("All tests passed!")


if __name__ == "__main__":
    test1()
