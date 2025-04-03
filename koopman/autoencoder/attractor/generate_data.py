import os

import numpy as np
import torch
from scipy.ndimage import gaussian_filter1d
from spatialmath.base import angle_wrap

from koopman.simulation.simulator import simulate, simulate_batch
from koopman.simulation.systems import DynamicalSystem, NonlinearAttractor2D, Pendulum

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))

attractor = NonlinearAttractor2D(NonlinearAttractor2D.Params(mu=2.0, lam=0.5))

if __name__ == "__main__":
    N = 1_000
    x0 = np.random.uniform(-10, 10, (N, 2))
    tf = 8.0
    dt = 0.05

    ts, xhist, uhist = simulate_batch(sys=attractor, tf=tf, dt=dt, x0=x0, u=lambda t, x: np.zeros((N, 1)))

    print("Saving data...")
    print("\tts:", ts.shape)
    print("\txhist:", xhist.shape)
    print("\tuhist:", uhist.shape)

    torch.save(
        {
            "ts": torch.as_tensor(ts),
            "xhist": torch.as_tensor(xhist),
            "uhist": torch.as_tensor(uhist),
            "tf": tf,
            "dt": dt
        }, os.path.join(SCRIPT_DIR, "attractor_data.pt"))
