import time
from typing import Any, Callable, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import solve_continuous_are
from tqdm import tqdm

from koopman.simulation.animation import PlotEnvironment
from koopman.simulation.systems import DynamicalSystem, Pendulum


def rk4_step(f: Callable, x: np.ndarray, u: np.ndarray, dt: float) -> np.ndarray:
    k1 = f(x, u)
    k2 = f(x + 0.5 * dt * k1, u)
    k3 = f(x + 0.5 * dt * k2, u)
    k4 = f(x + dt * k3, u)

    return x + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)


def simulate_batch(sys: DynamicalSystem,
                   tf: float,
                   dt: float,
                   u: Union[Callable, np.ndarray],
                   x0: np.ndarray,
                   obs_fn: Callable = lambda i, x_hist: x_hist[:, i, :],
                   pbar=True) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    N, nx = x0.shape

    ts = np.arange(0, tf + dt, dt)
    x_hist = np.zeros((N, len(ts), nx))

    if isinstance(u, np.ndarray):
        u_hist = u
    else:
        assert u(0.0, obs_fn(0, x_hist)).shape == (N, sys.nu), "Control function must return an array of shape (N, nu)"
        u_hist = np.zeros((N, len(ts) - 1, sys.nu))

    assert nx == sys.nx, "Initial states must have shape (N, nx)"
    x_hist[:, 0, :] = x0

    if pbar:
        iterator = tqdm(ts[:-1], desc="Simulation progress", total=len(ts) - 1)
    else:
        iterator = ts[:-1]

    for i, t in enumerate(iterator):
        observation = obs_fn(i, x_hist)

        if not isinstance(u, np.ndarray):
            u_hist[:, i, :] = u(t, observation)

        # Run one RK4 integration step
        x_hist[:, i + 1, :] = rk4_step(sys.dynamics, x_hist[:, i, :], u_hist[:, i, :], dt)

        # # Project state if necessary to keep it in the manifold
        # x_hist[:, i + 1, :] = sys.project_state(x_hist[:, i + 1, :])

    return ts, x_hist, u_hist


def simulate(sys: DynamicalSystem,
             tf: float,
             dt: float,
             u: Union[Callable, np.ndarray],
             x0: np.ndarray,
             log: bool = True,
             obs_fn: Callable = lambda i, x_hist: x_hist[i, :],
             pbar=True) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if isinstance(u, np.ndarray):
        u_modified = np.expand_dims(u, axis=0)
    else:
        u_modified = lambda t, x: np.expand_dims(u(t, np.squeeze(x, axis=0)), axis=0)

    obs_fn_modified = lambda i, x_hist: np.expand_dims(obs_fn(i, np.squeeze(x_hist, axis=0)), axis=0)
    ts, x_hist, u_hist = simulate_batch(sys, tf, dt, u_modified, np.expand_dims(x0, axis=0), obs_fn_modified, pbar)

    if log:
        sys.set_history(ts, np.squeeze(x_hist, axis=0), np.squeeze(u_hist, axis=0))

    return ts, np.squeeze(x_hist, axis=0), np.squeeze(u_hist, axis=0)


def simulate_pendulum():
    sys = Pendulum(Pendulum.Params(m=1, l=1, g=9.81))

    x0 = np.array([1.0, 0.0])
    tf = 5.0

    ts, xhist, uhist = simulate(sys, tf=tf, dt=0.02, u=lambda t, x: np.array([0.0]), x0=x0, log=True)

    # Render the simulation
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.grid(True)
    ax.set_aspect('equal')

    env = PlotEnvironment(fig, ax)
    env.add_element(Pendulum.PlotElement(env, sys))
    _ = env.render(t_range=(0, tf), fps=30)

    plt.show()


if __name__ == "__main__":
    simulate_pendulum()
