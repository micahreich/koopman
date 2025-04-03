import os
import random

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
from koopman.simulation.systems import NonlinearAttractor2D

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))


def simulate_with_observables(model: KoopmanAutoencoder, x0: torch.Tensor, uhist: torch.Tensor, device):
    assert x0.dim() == 1, "x0 must be a 1D tensor"
    assert uhist.dim() == 2, "u must be a 2D tensor"

    x0 = x0.to(device)
    uhist = uhist.to(device)

    Tm1, _ = uhist.shape

    xhist_pred = torch.empty(Tm1 + 1, model.nx).to(device)
    zhist = torch.empty(Tm1 + 1, model.nz).to(device)

    z_jm1 = model.forward(x0.unsqueeze(0))
    xhist_pred[0, :] = torch.squeeze(model.project(z_jm1), dim=0)
    zhist[0, :] = torch.squeeze(z_jm1, dim=0)

    for j in range(1, Tm1 + 1):
        z_jm1 = model.predict_z_next(z_jm1, uhist[j - 1, :].unsqueeze(0))

        xhist_pred[j, :] = torch.squeeze(model.project(z_jm1), dim=0)
        zhist[j, :] = torch.squeeze(z_jm1, dim=0)

    return xhist_pred, zhist


def evaluate(model: KoopmanAutoencoder, dataset: KoopmanDataset, epoch_idx: int, device):
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
    colors = list(mcolors.TABLEAU_COLORS.values())

    for i in range(NonlinearAttractor2D.nx):
        color = random.choice(colors)

        ax[0].plot(ts, xhist[:, i], color=color, linestyle='--', label=rf'$x_{i}$ True')
        ax[0].plot(ts, xhist_pred[:, i], color=color, label=rf'$x_{i}$ Pred')

    for i in range(NonlinearAttractor2D.nu):
        color = random.choice(colors)

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


def train(model: KoopmanAutoencoder, dataset: KoopmanDataset, n_epochs: int, batch_size: int, learning_rate: float):
    os.makedirs(os.path.join(SCRIPT_DIR, "eval_plots"), exist_ok=True)

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    # optimizer = torch.optim.Adam([
    #     {'params': model.param_groups['observables'], 'lr': learning_rate},
    #     {'params': model.param_groups['dynamics'], 'lr': learning_rate * 1e-10}
    # ])

    scheduler = CosineAnnealingLR(optimizer, T_max=n_epochs // 4, eta_min=1e-6)

    # Test forward pass of the model and loss computation, backprop call
    _xhist, _uhist = next(iter(dataloader))
    _xhist = _xhist.to(device)
    _uhist = _uhist.to(device)
    _xhist_flat = einops.rearrange(_xhist, 'b h nx -> (b h) nx')
    _uhist_flat = einops.rearrange(_uhist, 'b h nu -> (b h) nu')

    _zhist_flat = model.forward(_xhist_flat)
    _loss, _loss_by_parts = model.loss(_xhist_flat, _uhist_flat, _zhist_flat)
    _loss.backward()

    optimizer.zero_grad()

    # Begin training
    model.train()

    print("Beginning training... info:")
    print(f"\tBatch size: {batch_size}")
    print(f"\tLearning rate: {learning_rate}")
    print(f"\tEpochs: {n_epochs}")
    print(f"\tDevice: {device}")
    print(f"\tN params: {sum(p.numel() for p in model.parameters())}")

    loss_history = []
    loss_by_parts_history = {k: [] for k in _loss_by_parts.keys()}

    with torch.no_grad():
        evaluate(model, dataset, '_init', device)

    try:
        for epoch in range(n_epochs):
            pbar = tqdm(dataloader, desc=f"Epoch {epoch}", unit="batch")
            epoch_loss = 0.0

            for step, batch in enumerate(pbar):
                xhist, uhist = map(lambda x: x.to(device), batch)

                # xhist.requires_grad_(True)
                # uhist.requires_grad_(True)

                # B, _, _ = xhist.shape

                # Flatten the batch dimension
                xhist_flat = KoopmanDataset.flatten_batch(xhist)
                uhist_flat = KoopmanDataset.flatten_batch(uhist)

                zhist_flat = model.forward(xhist_flat)

                x0, u0, x_horizon, u_horizon = batch
                z0 = model.forward(x0)

                x_horizon_flat = model.flatten_batch(x_horizon)
                z_horizon_flat = model.forward(x_horizon_flat)
                z_horizon = model.unflatten_batch(z_horizon_flat)

                # zhist = einops.rearrange(zhist, '(b h) nz -> b h nz', b=B)

                # loss_total, _loss_parts = model.loss(xhist_flat, uhist_flat, zhist_flat)
                loss_total, _loss_parts = model.loss(
                    xs=(x0, x_horizon),
                    us=(u0, u_horizon),
                    zs=(z0, z_horizon),
                )

                optimizer.zero_grad()
                loss_total.backward()
                optimizer.step()

                epoch_loss += loss_total.item()
                current_lr = optimizer.param_groups[0]['lr']
                pbar.set_postfix(loss=f"{loss_total.item() :.6f}", lr=f"{current_lr:.6f}")

                loss_history.append(loss_total.item())
                for k, v in _loss_parts.items():
                    loss_by_parts_history[k].append(v.item())

            scheduler.step()
            epoch_loss /= len(dataloader)

            with torch.no_grad():
                evaluate(model, dataset, epoch, device)

    except KeyboardInterrupt:
        print("Training interrupted")

    # Plot the loss history
    fig, ax = plt.subplots(figsize=(10, 5))

    xs = np.arange(len(loss_history)) / len(dataloader)
    for k, v in loss_by_parts_history.items():
        ax.plot(xs, v, label=k, alpha=0.5, linewidth=0.5)

    ax.plot(xs, loss_history, label="total", color='black', linewidth=0.5)

    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.1))

    ax.set_title("Loss over epochs")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.grid(True)
    ax.legend()
    plt.savefig(os.path.join(SCRIPT_DIR, "loss_plot.png"))
    plt.close(fig)


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
        H=pred_horizon,
        params_init='eye',
        hidden_dims=[32, 32],
        activation=nn.Mish,
        use_layernorm=False,
        horizon_loss_weight=5.0,
        L1_loss_weight=0.5,
    )

    train(model, dataset, n_epochs=100, batch_size=64, learning_rate=0.001)

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
