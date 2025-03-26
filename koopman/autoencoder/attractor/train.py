import os
import random

import einops
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from koopman import utils
from koopman.autoencoder.dataset import KoopmanDataset
from koopman.autoencoder.model import KoopmanAutoencoder
from koopman.simulation.systems import NonlinearAttractor2D

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))


def evaluate(model: KoopmanAutoencoder, xhist: torch.Tensor, uhist: torch.Tensor, ts: torch.Tensor, epoch_idx: int,
             device):
    N, T, _ = xhist.shape
    i_rand = torch.randint(0, N, (1, )).item()

    xhist_i = xhist[i_rand, :, :].to(device)
    uhist_i = uhist[i_rand, :, :].to(device)
    xhist_i_pred = torch.empty_like(xhist_i).to(device)

    x0 = xhist_i[0, :]
    z_jm1 = model.forward(x0.unsqueeze(0))
    xhist_i_pred[0, :] = torch.squeeze(model.project(z_jm1), dim=0)

    for j in range(1, T):
        z_jm1 = model.predict_z_next(z_jm1, uhist_i[j - 1, :].unsqueeze(0))
        xhist_i_pred[j, :] = torch.squeeze(model.project(z_jm1), dim=0)

    # Create a plot of the state evolution prediction over time vs. true state
    xhist_i = utils.torch_to_numpy(xhist_i)
    uhist_i = utils.torch_to_numpy(uhist_i)
    xhist_i_pred = utils.torch_to_numpy(xhist_i_pred)
    ts = utils.torch_to_numpy(ts)

    fig, ax = plt.subplots(2, 1, figsize=(10, 8))
    colors = list(mcolors.TABLEAU_COLORS.values())

    for i in range(NonlinearAttractor2D.nx):
        color = random.choice(colors)

        ax[0].plot(ts, xhist_i[:, i], color=color, linestyle='--', label=rf'$x_{i}$ True')
        ax[0].plot(ts, xhist_i_pred[:, i], color=color, label=rf'$x_{i}$ Pred')

    for i in range(NonlinearAttractor2D.nu):
        color = random.choice(colors)

        ax[1].plot(ts[:-1], uhist_i[:, i], color=color, linestyle='--', label=rf'$u_{i}$')

    ax[0].legend()
    ax[1].legend()
    ax[0].grid(True)
    ax[1].grid(True)

    fig.suptitle(f"Epoch {epoch_idx} - State Prediction vs. True State")
    plt.savefig(os.path.join(SCRIPT_DIR, "eval_plots", f"attractor_epoch_{epoch_idx}.png"))
    plt.close(fig)


def train(model: KoopmanAutoencoder, dataset: KoopmanDataset, n_epochs: int, batch_size: int, learning_rate: float):
    os.makedirs(os.path.join(SCRIPT_DIR, "eval_plots"), exist_ok=True)

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = CosineAnnealingLR(optimizer, T_max=n_epochs // 4, eta_min=1e-6)

    # Test forward pass of the model and loss computation, backprop call
    _xhist, _uhist = next(iter(dataloader))
    _xhist = _xhist.to(device)
    _uhist = _uhist.to(device)

    _zhist = model.forward(_xhist)
    _loss = model.loss(_xhist, _uhist, _zhist)
    _loss.backward()

    optimizer.zero_grad()

    # Begin training
    with torch.no_grad():
        evaluate(model, dataset.x_hist, dataset.u_hist, dataset.ts, 'init', device)

    for epoch in range(n_epochs):
        pbar = tqdm(dataloader, desc=f"Epoch {epoch}", unit="batch")
        epoch_loss = 0.0

        for step, batch in enumerate(pbar):
            xhist, uhist = batch
            xhist = xhist.to(device)
            uhist = uhist.to(device)

            B, _, _ = xhist.shape

            # Flatten the batch dimension
            xhist_flat = einops.rearrange(xhist, 'b h nx -> (b h) nx')

            zhist = model.forward(xhist_flat)
            zhist = einops.rearrange(zhist, '(b h) nz -> b h nz', b=B)

            loss = model.loss(xhist, uhist, zhist)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            current_lr = optimizer.param_groups[0]['lr']
            pbar.set_postfix(loss=f"{loss.item() :.6f}", lr=f"{current_lr:.6f}")

        scheduler.step()
        epoch_loss /= len(dataloader)

        with torch.no_grad():
            evaluate(model, dataset.x_hist, dataset.u_hist, dataset.ts, epoch, device)


if __name__ == "__main__":
    pred_horizon = 10

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
        H=pred_horizon,
        params_init='eye',
        hidden_dims=[32, 32],
        activation=nn.Mish,
        use_layernorm=False,
        projection_loss_weight=2.0,
        horizon_loss_weight=5.0,
        L1_loss_weight=0.1,
    )

    train(model, dataset, n_epochs=100, batch_size=64, learning_rate=0.001)
