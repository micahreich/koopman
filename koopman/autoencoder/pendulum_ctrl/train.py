import os
import random
from typing import Union

import einops
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import torch
from torch import nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from koopman import utils
from koopman.autoencoder.dataset import KoopmanDataset
from koopman.autoencoder.model import KoopmanAutoencoder
from koopman.autoencoder.training_loop import simulate_with_observables, train
from koopman.simulation.systems import Pendulum

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))


def evaluate(model: KoopmanAutoencoder, dataset: KoopmanDataset, epoch_idx: Union[str, int], device: torch.device):
    is_training = model.training
    model.eval()

    xhist, uhist, ts = dataset.sample_eval_trajectories(1)
    x0 = xhist[0, :]
    xhist_pred, _ = simulate_with_observables(model, x0, uhist, device)

    # if dataset.normalize:
    #     xhist_pred = dataset.stats.denormalize_x(xhist_pred)

    # Create a plot of the state evolution prediction over time vs. true state
    xhist = utils.torch_to_numpy(xhist)
    uhist = utils.torch_to_numpy(uhist)
    xhist_pred = utils.torch_to_numpy(xhist_pred)
    ts = utils.torch_to_numpy(ts)

    fig, ax = plt.subplots(2, 1, figsize=(10, 8))
    cmap = plt.get_cmap("viridis")
    colors = [cmap(i) for i in np.linspace(0, 1, Pendulum.nx + Pendulum.nu)]

    for i in range(Pendulum.nx):
        color = colors[i]

        ax[0].plot(ts, xhist[:, i], color=color, linestyle='--', label=rf'$x_{i}$ True')
        ax[0].plot(ts, xhist_pred[:, i], color=color, label=rf'$x_{i}$ Pred')

    for i in range(Pendulum.nu):
        color = colors[i + Pendulum.nx]

        ax[1].plot(ts[:-1], uhist[:, i], color=color, linestyle='--', label=rf'$u_{i}$')

    ax[0].legend()
    ax[1].legend()
    ax[0].grid(True)
    ax[1].grid(True)

    fig.suptitle(f"Epoch {epoch_idx} - State Prediction vs. True State")
    plt.savefig(os.path.join(SCRIPT_DIR, "eval_plots", f"pendulum_epoch_{epoch_idx}.png"))
    plt.close(fig)

    if is_training:
        model.train()

    return xhist, uhist, xhist_pred


if __name__ == "__main__":
    pred_horizon = 50

    data = torch.load(os.path.join(SCRIPT_DIR, "pendulum_data.pt"), weights_only=False)
    ts = data["ts"]
    xhist = data["xhist"]
    uhist = data["uhist"]
    dt = data["dt"]
    stats = data["stats"]

    print(f"Dataset info...")
    print(f"\txhist shape: {xhist.shape}")
    print(f"\tuhist shape: {uhist.shape}")
    print(f"\ttime steps: {ts.shape}")
    print(f"\tdt: {dt}")

    dataset = KoopmanDataset(xhist, uhist, ts, stats, pred_horizon, normalize=True)

    model = KoopmanAutoencoder(
        nx=Pendulum.nx,
        nu=Pendulum.nu,
        nz=32,
        H=pred_horizon,
        params_init='eye',
        hidden_dims=[128, 128, 128],
        activation=nn.Mish,
        use_layernorm=False,
        horizon_loss_weight=10.0,
        L1_reg_weight=0.0,
        jacobian_reg_weight=0.0,
    )

    train(model, dataset, n_epochs=100, batch_size=128, learning_rate=1e-3, evaluate=evaluate, save_dir=SCRIPT_DIR)
