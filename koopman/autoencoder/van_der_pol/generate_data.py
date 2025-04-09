import os
import shutil

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import torch
from numpy.polynomial.chebyshev import Chebyshev
from scipy.ndimage import gaussian_filter1d
from sklearn.preprocessing import PolynomialFeatures
from spatialmath.base import angle_wrap
from torch import nn

from koopman import utils
from koopman.autoencoder.dataset import KoopmanDataset, KoopmanDatasetStats
from koopman.autoencoder.model import KoopmanAutoencoder
from koopman.autoencoder.training_loop import simulate_with_observables, train
from koopman.edmd.edmd import eDMD
from koopman.simulation.simulator import simulate, simulate_batch
from koopman.simulation.systems import DynamicalSystem, VanDerPolOscillator

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))

if __name__ == "__main__":
    # Run a simulation with the Van Der Pol system
    vanderpol = VanDerPolOscillator()

    N = 500
    x0 = np.random.uniform(-1, 1, (N, 2))
    tf = 10.0
    dt = 0.05
    T = int(tf / dt)
    # controls = np.random.randn(N, T, 1) * 5.0
    controls = np.zeros((N, T, 1))  #np.random.randn(N, T, 1) * 5.0

    ts, xhist, uhist = simulate_batch(sys=vanderpol, tf=tf, dt=dt, u=controls, x0=x0)

    print(f"Simulation finished. {xhist.shape=}, {uhist.shape=}")

    xhist = torch.as_tensor(xhist, dtype=torch.float32)
    uhist = torch.as_tensor(uhist, dtype=torch.float32)
    ts = torch.as_tensor(ts, dtype=torch.float32)
    stats = KoopmanDatasetStats.from_tensors(xhist, uhist, dt)

    torch.save({
        "ts": ts,
        "xhist": xhist,
        "uhist": uhist,
        "tf": tf,
        "dt": dt,
        "stats": stats,
    }, os.path.join(SCRIPT_DIR, "pendulum_data.pt"))
