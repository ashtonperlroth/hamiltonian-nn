"""Training loop for HNN and baseline models.

Can be run as a script:
    python -m src.train --system spring
    python -m src.train --system pendulum
    python -m src.train --system twobody
"""

import argparse
import os

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.baseline import BaselineMLP
from src.hnn import HNN
from src.systems import SYSTEMS, generate_trajectories


def make_dataset(system_name, n_trajectories=50, noise_std=0.0, seed=42):
    """Generate training data and return as TensorDatasets."""
    data = generate_trajectories(
        system_name,
        n_trajectories=n_trajectories,
        t_span=(0, 20),
        dt=0.05,
        noise_std=noise_std,
        rng=seed,
    )
    states = torch.tensor(data["states"], dtype=torch.float32)
    derivs = torch.tensor(data["derivatives"], dtype=torch.float32)

    # 80/20 train/test split
    n = len(states)
    idx = np.random.default_rng(seed).permutation(n)
    split = int(0.8 * n)
    train_idx, test_idx = idx[:split], idx[split:]

    train_ds = TensorDataset(states[train_idx], derivs[train_idx])
    test_ds = TensorDataset(states[test_idx], derivs[test_idx])
    return train_ds, test_ds


def train_model(model, train_ds, test_ds, epochs=2000, lr=1e-3, batch_size=256):
    """Train a model and return loss history."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    loss_fn = nn.MSELoss()

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size)

    history = {"train_loss": [], "test_loss": []}

    for epoch in range(epochs):
        # Train
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        for x, y in train_loader:
            pred = model(x)
            loss = loss_fn(pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1
        scheduler.step()

        # Evaluate
        model.eval()
        test_loss = 0.0
        n_test = 0
        with torch.no_grad():
            for x, y in test_loader:
                # HNN needs grad, so handle differently
                if isinstance(model, HNN):
                    with torch.enable_grad():
                        pred = model(x)
                else:
                    pred = model(x)
                test_loss += loss_fn(pred, y).item()
                n_test += 1

        train_avg = epoch_loss / n_batches
        test_avg = test_loss / max(n_test, 1)
        history["train_loss"].append(train_avg)
        history["test_loss"].append(test_avg)

        if (epoch + 1) % 200 == 0:
            print(
                f"  Epoch {epoch + 1:4d} | "
                f"train: {train_avg:.6f} | test: {test_avg:.6f}"
            )

    return history


def train_system(system_name, epochs=2000, save_dir="models"):
    """Train both HNN and baseline on a system and save weights."""
    print(f"\n{'='*60}")
    print(f"Training on: {system_name}")
    print(f"{'='*60}")

    sys_info = SYSTEMS[system_name]
    state_dim = sys_info["state_dim"]

    train_ds, test_ds = make_dataset(system_name)
    print(f"Data: {len(train_ds)} train, {len(test_ds)} test samples")

    # Train HNN
    print("\n--- HNN ---")
    hnn = HNN(state_dim)
    hnn_history = train_model(hnn, train_ds, test_ds, epochs=epochs)

    # Train Baseline
    print("\n--- Baseline MLP ---")
    baseline = BaselineMLP(state_dim)
    bl_history = train_model(baseline, train_ds, test_ds, epochs=epochs)

    # Save models
    os.makedirs(save_dir, exist_ok=True)
    torch.save(hnn.state_dict(), os.path.join(save_dir, f"hnn_{system_name}.pt"))
    torch.save(
        baseline.state_dict(), os.path.join(save_dir, f"baseline_{system_name}.pt")
    )

    return {
        "hnn": hnn,
        "baseline": baseline,
        "hnn_history": hnn_history,
        "baseline_history": bl_history,
    }


def main():
    parser = argparse.ArgumentParser(description="Train HNN and baseline models")
    parser.add_argument(
        "--system",
        type=str,
        default="spring",
        choices=list(SYSTEMS.keys()),
        help="Physical system to train on",
    )
    parser.add_argument("--epochs", type=int, default=2000)
    args = parser.parse_args()

    train_system(args.system, epochs=args.epochs)


if __name__ == "__main__":
    main()
