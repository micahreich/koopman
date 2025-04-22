import os
import random
import shutil
from typing import Callable, Optional

import einops
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import torch
from torch import nn
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from koopman import utils
from koopman.autoencoder.dataset import KoopmanDataset
from koopman.autoencoder.model import KoopmanAutoencoder


def simulate_with_observables(model: KoopmanAutoencoder, x0: torch.Tensor, uhist: torch.Tensor, device):
    assert x0.dim() == 1, "x0 must be a 1D tensor"
    assert uhist.dim() == 2, "u must be a 2D tensor"

    x0 = x0.to(device)
    uhist = uhist.to(device)

    Tm1, _ = uhist.shape

    xhist_pred = torch.empty(Tm1 + 1, model.nx).to(device)
    zhist = torch.empty(Tm1 + 1, model.nz).to(device)

    z_jm1 = model.forward(x0)
    xhist_pred[0, :] = model.project(z_jm1)
    zhist[0, :] = z_jm1

    A, B, Cs = model.get_dynamics()

    for j in range(1, Tm1 + 1):
        z_jm1 = model.predict_z_next(z_jm1, uhist[j - 1, :], A, B, Cs)

        xhist_pred[j, :] = model.project(z_jm1)
        zhist[j, :] = z_jm1

    return xhist_pred, zhist


def test_forward_pass(model: KoopmanAutoencoder, dataloader: DataLoader, device):
    batch = next(iter(dataloader))
    x0, u0, x_horizon, u_horizon = map(lambda x: x.to(device), batch)
    batch_size = x0.shape[0]
    pred_horizon = x_horizon.shape[1]

    z0 = model.forward(x0)
    x_horizon_flat = model.flatten_batch(x_horizon)
    z_horizon_flat = model.forward(x_horizon_flat)
    z_horizon = model.unflatten_batch(batch_size, z_horizon_flat)
    loss_total, _loss_by_parts = model.loss(xs=(x0, x_horizon),
                                            us=(u0, u_horizon),
                                            zs=(z0, z_horizon),
                                            pred_horizon=pred_horizon)

    loss_total.backward()
    return loss_total, _loss_by_parts


# def generate_curriculum(ph_max: int, ph_min: int, n_epochs):
#     warmup_ratio = 0.2

#     n_warmup_epochs = int(warmup_ratio * n_epochs)
#     y = np.linspace(ph_min, ph_max, n_epochs - n_warmup_epochs, dtype=int)

#     ph_schedule = np.concatenate((np.full(n_warmup_epochs, ph_min), y))
#     return ph_schedule


def train(model: KoopmanAutoencoder,
          dataset: KoopmanDataset,
          n_epochs: int,
          batch_size: int,
          learning_rate: float,
          evaluate: Optional[Callable] = None,
          save_dir: Optional[str] = ""):
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # # Experiment with different learning rates for the dynamics and observables
    # optimizer = torch.optim.Adam([{
    #     'params': model.param_groups['observables'],
    #     'lr': learning_rate
    # }, {
    #     'params': model.param_groups['dynamics'],
    #     'lr': learning_rate
    # }])

    # scheduler = CosineAnnealingLR(optimizer, T_max=n_epochs // 4, eta_min=1e-6)
    scheduler = StepLR(optimizer, step_size=15, gamma=0.5)

    # Test forward pass of the model and loss computation, backprop call
    _, _loss_by_parts = test_forward_pass(model, dataloader, device)
    optimizer.zero_grad()

    # Begin training
    print("Beginning training...")
    print(f"    Dataset info:")
    print(f"\txhist shape: {dataset.x_hist.shape}")
    print(f"\tuhist shape: {dataset.u_hist.shape}")
    print(f"\tdt: {dataset.dt}")
    print(f"    Training info:")
    print(f"\tBatch size: {batch_size}")
    print(f"\tLearning rate: {learning_rate}")
    print(f"\tEpochs: {n_epochs}")
    print(f"\tDevice: {device}")
    print(f"\tNum params: {sum(p.numel() for p in model.parameters())}")

    loss_history = []
    loss_by_parts_history = {k: [] for k in _loss_by_parts.keys()}

    # pred_horizon_schedule = generate_curriculum(ph_max=pred_horizon_max, ph_min=pred_horizon_min, n_epochs=n_epochs)

    if evaluate is not None:
        with torch.no_grad():
            model.eval()
            evaluate(model, dataset, '_init', device)

    try:
        for epoch in range(n_epochs):
            model.train()
            pbar = tqdm(dataloader, desc=f"Epoch {epoch}", unit="batch")
            epoch_loss = 0.0

            for step, batch in enumerate(pbar):
                # Move the batch to the model's device and set requires_grad
                # for Jacobian regularization
                x0, u0, x_horizon, u_horizon = map(lambda x: x.to(device), batch)
                B = x0.shape[0]
                pred_horizon = x_horizon.shape[1]

                z0 = model.forward(x0)

                x_horizon_flat = model.flatten_batch(x_horizon)
                z_horizon_flat = model.forward(x_horizon_flat)
                z_horizon = model.unflatten_batch(B, z_horizon_flat)

                loss_total, _loss_parts = model.loss(xs=(x0, x_horizon),
                                                   us=(u0, u_horizon),
                                                   zs=(z0, z_horizon),
                                                   pred_horizon=pred_horizon)

                optimizer.zero_grad()
                loss_total.backward()
                optimizer.step()

                epoch_loss += loss_total.item()
                current_lr = optimizer.param_groups[0]['lr']
                pbar.set_postfix(loss=f"{loss_total.item() :.6f}", lr=f"{current_lr:.6f}")

                loss_history.append(loss_total.item())
                for k, v in _loss_parts.items():
                    loss_by_parts_history[k].append(v.item())

            # scheduler.step()
            epoch_loss /= len(dataloader)

            with torch.no_grad():
                model.eval()
                evaluate(model, dataset, epoch, device)

    except KeyboardInterrupt:
        print("Training interrupted")

    torch.save(model.state_dict(), os.path.join(save_dir, "model.pth"))

    # Plot the loss history
    fig, ax = plt.subplots(figsize=(10, 5))

    xs = np.arange(len(loss_history)) / len(dataloader)
    for k, v in loss_by_parts_history.items():
        ax.plot(xs, v, label=k, alpha=0.5, linewidth=0.5)

    ax.plot(xs, loss_history, label="total", color='black', linewidth=0.5)

    ax.xaxis.set_major_locator(ticker.MultipleLocator((epoch + 1) // 5))
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(1))

    ax.set_title("Loss over epochs")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.grid(True)
    ax.legend()
    plt.savefig(os.path.join(save_dir, "loss_plot.png"))
    plt.close(fig)


# if __name__ == "__main__":
#     phs = generate_curriculum(ph_max=30, ph_min=10, n_epochs=101)
#     print("Prediction horizons:", phs)
