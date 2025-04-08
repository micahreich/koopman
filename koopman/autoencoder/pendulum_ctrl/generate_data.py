import os

import numpy as np
import torch
from scipy.ndimage import gaussian_filter1d
from spatialmath.base import angle_wrap

from koopman.autoencoder.dataset import KoopmanDatasetStats
from koopman.simulation.simulator import simulate, simulate_batch
from koopman.simulation.systems import Pendulum

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))

pendulum = Pendulum(Pendulum.Params(m=1, l=1, g=9.81, b=0.5))


def generate_gaussian_smooth_controls(T, N=1, nu=1, std=1.0, sigma_range=[0.5, 2.0]):
    u_raw = np.random.randn(N, T, nu) * std
    u_smooth = np.zeros_like(u_raw)

    sigmas = np.random.uniform(sigma_range[0], sigma_range[1], size=(N, ))

    # Apply gaussian_filter1d over time axis (axis=1) for each control dimension
    for n in range(N):
        for i in range(nu):
            u_smooth[n, :, i] = gaussian_filter1d(u_raw[n, :, i], sigma=sigmas[n], mode="nearest")

    return u_smooth


def generate_decaying_sinusoid_controls(T, N, nu=1):
    freq_range = (1.0, 5.0)
    decay_range = (0.1, 10.0)
    amp_range = (1.0, 10.0)
    phase_range = (0, 2 * np.pi)

    t = np.linspace(0, 1, T)  # time over 1 second

    # Random parameters per sample per channel
    freqs = np.random.uniform(*freq_range, size=(N, nu))
    decays = np.random.uniform(*decay_range, size=(N, nu))
    amps = np.random.uniform(*amp_range, size=(N, nu))
    phases = np.random.uniform(*phase_range, size=(N, nu))

    # Shape broadcasting to (N, T, nu)
    t = t[None, :, None]  # (1, T, 1)

    freqs = freqs[:, None, :]  # (N, 1, nu)
    decays = decays[:, None, :]
    amps = amps[:, None, :]
    phases = phases[:, None, :]

    signals = amps * np.exp(-decays * t) * np.sin(2 * np.pi * freqs * t + phases)
    return signals


if __name__ == "__main__":
    N = 1_000

    theta_0 = np.random.uniform(-np.pi, np.pi, (N, 1))
    omega_0 = np.random.uniform(-5, 5, (N, 1))
    x0 = np.hstack((theta_0, omega_0))

    tf = 8.0
    dt = 0.05
    T = int(tf / dt)

    controls = generate_decaying_sinusoid_controls(T=T, N=N, nu=Pendulum.nu)

    def u(t, x):
        # Generate a smooth control trajectory
        t_idx = int(t / dt)
        return controls[:, t_idx, :]

    ts, xhist, uhist = simulate_batch(sys=pendulum, tf=tf, dt=dt, x0=x0, u=u)

    xhist = torch.as_tensor(xhist, dtype=torch.float32)
    uhist = torch.as_tensor(uhist, dtype=torch.float32)
    ts = torch.as_tensor(ts, dtype=torch.float32)
    stats = KoopmanDatasetStats.from_tensors(xhist, uhist, dt)

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
