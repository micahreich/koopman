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

if __name__ == "__main__":
    N = 5_000

    theta_0 = np.random.uniform(-np.pi, np.pi, (N, 1))
    omega_0 = np.random.uniform(-5, 5, (N, 1))
    x0 = np.hstack((theta_0, omega_0))

    tf = 8.0
    dt = 0.1

    ts, xhist, uhist = simulate_batch(sys=pendulum, tf=tf, dt=dt, x0=x0, u=lambda t, x: np.zeros((N, 1)))

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
