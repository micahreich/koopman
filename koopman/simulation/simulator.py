import time
from fractions import Fraction
from typing import Any, Callable, Tuple, Union

import jax
import jax.numpy as jnp
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


def get_time_steps(tf: float, dt: float) -> np.ndarray:
    dt_frac = Fraction.from_float(dt).limit_denominator()
    tf_frac = Fraction.from_float(tf).limit_denominator()
    num_steps = int(tf_frac / dt_frac)

    return dt * np.arange(num_steps + 1)


# def simulate_batch(sys: DynamicalSystem,
#                    tf: float,
#                    dt: float,
#                    u: Union[Callable, np.ndarray],
#                    x0: np.ndarray,
#                    pbar=True) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
#     ts = get_time_steps(tf, dt)

#     if pbar:
#         iterator = tqdm(ts[:-1], desc="Simulation progress", total=len(ts) - 1)
#     else:
#         iterator = ts[:-1]

#     x_hist = np.zeros((len(ts), sys.nx))
#     u_hist = np.zeros((len(ts) - 1, sys.nu))

#     x_hist[0] = x0

#     for i, t_frac in enumerate(iterator):
#         t = float(t_frac)

#         if not isinstance(u, np.ndarray):
#             u_hist[i] = u(t, x_hist[i])

#         # Run one RK4 integration step
#         x_hist[i + 1] = rk4_step(sys.dynamics, x_hist[i], u_hist[i], dt)

#     return ts.astype(np.float32), x_hist, u_hist

# def simulate(sys: DynamicalSystem,
#              tf: float,
#              dt: float,
#              u: Union[Callable, np.ndarray],
#              x0: np.ndarray,
#              log: bool = True,
#              pbar=True) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
#     ts = get_time_steps(tf, dt)

#     if pbar:
#         iterator = tqdm(ts[:-1], desc="Simulation progress", total=len(ts) - 1)
#     else:
#         iterator = ts[:-1]

#     if x0.ndim == 1:
#         x_hist = np.zeros((len(ts), sys.nx))
#         u_hist = np.zeros((len(ts) - 1, sys.nu))
#     else:
#         N = x0.shape[0]
#         x_hist = np.zeros((len(ts), N, sys.nx))
#         u_hist = np.zeros((len(ts) - 1, N, sys.nu))

#     x_hist[0] = x0

#     for i, t_frac in enumerate(iterator):
#         t = float(t_frac)

#         if not isinstance(u, np.ndarray):
#             u_hist[i] = u(t, x_hist[i])

#         # Run one RK4 integration step
#         x_hist[i + 1] = rk4_step(sys.dynamics, x_hist[i], u_hist[i], dt)

#     return ts.astype(np.float32), x_hist, u_hist


def simulate(dynamics: Callable, tf: float, dt: float, u_fn: Callable[[float, jnp.ndarray], jnp.ndarray],
             x0: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    T = int(jnp.ceil(tf / dt))
    ts = dt * jnp.arange(0, T + 1)

    def step(x, t):
        u = u_fn(t, x)
        x_next = rk4_step(dynamics, x, u, dt)
        return x_next, (x, u)

    x0 = jnp.asarray(x0)
    _, (x_hist, u_hist) = jax.lax.scan(step, x0, ts[:-1])
    x_hist = jnp.concatenate([x0[None], x_hist], axis=0)  # prepend initial state

    return ts, x_hist, u_hist


def simulate_pendulum():
    sys = Pendulum(Pendulum.Params(m=1, l=1, g=9.81))

    x0 = np.array([1.0, 0.0])
    tf = 5.0

    ts, xhist, uhist = simulate(sys, tf=tf, dt=0.02, u=lambda t, x: np.array([0.0]), x0=x0, log=True)

    # # Render the simulation
    # fig, ax = plt.subplots(figsize=(5, 5))
    # ax.grid(True)
    # ax.set_aspect('equal')

    # env = PlotEnvironment(fig, ax)
    # env.add_element(Pendulum.PlotElement(env, sys))
    # _ = env.render(t_range=(0, tf), fps=30)

    # plt.show()


if __name__ == "__main__":
    simulate_pendulum()
