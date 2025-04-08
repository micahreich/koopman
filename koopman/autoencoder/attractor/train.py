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
from koopman.simulation.systems import NonlinearAttractor2D

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))


def evaluate(model: KoopmanAutoencoder, dataset: KoopmanDataset, epoch_idx: Union[str, int], device: torch.device):
    is_training = model.training
    model.eval()

    xhist, uhist, ts = dataset.sample_eval_trajectories(1)
    x0 = xhist[0, :]
    xhist_pred, _ = simulate_with_observables(model, x0, uhist, device)

    # Create a plot of the state evolution prediction over time vs. true state
    xhist = utils.torch_to_numpy(xhist)
    uhist = utils.torch_to_numpy(uhist)
    xhist_pred = utils.torch_to_numpy(xhist_pred)
    ts = utils.torch_to_numpy(ts)

    fig, ax = plt.subplots(2, 1, figsize=(10, 8))
    cmap = plt.get_cmap("viridis")
    colors = [cmap(i) for i in np.linspace(0, 1, NonlinearAttractor2D.nx + NonlinearAttractor2D.nu)]

    for i in range(NonlinearAttractor2D.nx):
        color = colors[i]

        ax[0].plot(ts, xhist[:, i], color=color, linestyle='--', label=rf'$x_{i}$ True')
        ax[0].plot(ts, xhist_pred[:, i], color=color, label=rf'$x_{i}$ Pred')

    for i in range(NonlinearAttractor2D.nu):
        color = colors[i + NonlinearAttractor2D.nx]

        ax[1].plot(ts[:-1], uhist[:, i], color=color, linestyle='--', label=rf'$u_{i}$')

    ax[0].legend()
    ax[1].legend()
    ax[0].grid(True)
    ax[1].grid(True)

    fig.suptitle(f"Epoch {epoch_idx} - State Prediction vs. True State")
    plt.savefig(os.path.join(SCRIPT_DIR, "eval_plots", f"attractor_epoch_{epoch_idx}.png"))
    plt.close(fig)

    if is_training:
        model.train()

    return xhist, uhist, xhist_pred


if __name__ == "__main__":
    pred_horizon = 20

    data = torch.load(os.path.join(SCRIPT_DIR, "attractor_data.pt"))
    ts = data["ts"]
    xhist = data["xhist"]
    uhist = data["uhist"]
    dt = data["dt"]

    dataset = KoopmanDataset(xhist, uhist, ts, pred_horizon, dt)

    model = KoopmanAutoencoder(
        nx=NonlinearAttractor2D.nx,
        nu=NonlinearAttractor2D.nu,
        nz=3,
        pred_horizon=pred_horizon,
        params_init='eye',
        hidden_dims=[32, 32],
        activation=nn.Mish,
        use_layernorm=False,
        horizon_loss_weight=10.0,
        L1_reg_weight=0.5,
        jacobian_reg_weight=0.0,
    )

    train(model, dataset, n_epochs=100, batch_size=64, learning_rate=0.001, evaluate=evaluate, save_dir=SCRIPT_DIR)

    # # Plot the observables history vs the ground-truth observables
    # model.eval()
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # xhist, uhist, ts = dataset.sample_eval_trajectories(1)
    # x0 = xhist[0, :]
    # xhist_pred, zhist = simulate_with_observables(model, x0, uhist, device)

    # xhist = utils.torch_to_numpy(xhist)
    # zhist = utils.torch_to_numpy(zhist)

    # ratio = xhist[:, 0]**2 / zhist[:, 2]

    # print(f"Ratio: {ratio}")
    # print(f"Ratio (mean): {np.mean(ratio)}")
    # print(f"Ratio (std): {np.std(ratio)}")
