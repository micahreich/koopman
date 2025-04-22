import os

import numpy as np
import torch
from scipy.ndimage import gaussian_filter1d
from spatialmath.base import angle_wrap
import matplotlib.pyplot as plt

from koopman.autoencoder.dataset import KoopmanDatasetStats
from koopman.simulation.simulator import simulate, simulate_batch
from koopman.simulation.systems import Pendulum

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))



def generate_gaussian_smooth_controls(T, N=1, nu=1, std=10.0, sigma_range=[0.5, 2.0]):
    u_raw = np.random.randn(N, T, nu) * std
    u_smooth = np.zeros_like(u_raw)

    sigmas = np.random.uniform(sigma_range[0], sigma_range[1], size=(N, ))

    # Apply gaussian_filter1d over time axis (axis=1) for each control dimension
    for n in range(N):
        for i in range(nu):
            u_smooth[n, :, i] = gaussian_filter1d(u_raw[n, :, i], sigma=sigmas[n], mode="nearest")

    return u_smooth

if __name__ == "__main__":
    N = 10_000

    theta_0 = np.random.uniform(-np.pi, np.pi, (N, 1))
    omega_0 = np.random.uniform(-5, 5, (N, 1))
    x0 = np.hstack((theta_0, omega_0))

    tf = 15.0
    dt = 0.1

    u = generate_gaussian_smooth_controls(int(tf / dt), N, std=10.0, sigma_range=[1, 3])
    ts, xhist, uhist = simulate_batch(sys=pendulum, tf=tf, dt=dt, x0=x0, u=u)

    # ts, xhist, uhist = simulate_batch(sys=pendulum, tf=tf, dt=dt, x0=x0, u=lambda t, x: np.zeros((N, 1)))

    xhist = torch.as_tensor(xhist, dtype=torch.float32)
    uhist = torch.as_tensor(uhist, dtype=torch.float32)
    ts = torch.as_tensor(ts, dtype=torch.float32)

    # plot sample trajectories
    n_samples = 5

    samples = np.random.randint(0, N, n_samples)
    fig, ax = plt.subplots(2, 1, figsize=(10, 8))
    cmap = plt.get_cmap("viridis")
    colors = [cmap(i) for i in np.linspace(0, 1, n_samples)]
    
    for sample_i, color in zip(samples, colors):
        ax[0].plot(ts, xhist[sample_i, :, 0], color=color, label=rf"$x_0$")
        ax[0].plot(ts, xhist[sample_i, :, 1], linestyle='--', color=color, label=rf"$x_1$")
        ax[1].plot(ts[:-1], uhist[sample_i, :, 0], color=color, label=rf"$x_1$")

    ax[0].legend()
    ax[1].legend()
    ax[0].grid(True)
    ax[1].grid(True)
    fig.suptitle("Sample trajectories")
    plt.savefig(os.path.join(SCRIPT_DIR, "generated_data_samples.png"))
    plt.close(fig)
    
    xhist_wrapped = torch.empty((xhist.shape[0], xhist.shape[1], 3), dtype=xhist.dtype)
    xhist_wrapped[..., 0] = torch.cos(xhist[..., 0])
    xhist_wrapped[..., 1] = torch.sin(xhist[..., 0])
    xhist_wrapped[..., 2] = xhist[..., 1]
    xhist = xhist_wrapped

    # do not normalize sin(th), cos(th)
    stats = KoopmanDatasetStats.from_tensors(xhist, uhist, dt, 
                                             normalize_indices=np.array([2]))

    print("Saving data...")
    print("\tts:", ts.shape)
    print("\txhist:", xhist.shape)
    print("\tuhist:", uhist.shape)
    print(stats)

    torch.save({
        "ts": ts,
        "xhist": xhist,
        "uhist": uhist,
        "tf": tf,
        "dt": dt,
        "stats": stats,
    }, os.path.join(SCRIPT_DIR, "pendulum_data.pt"))
