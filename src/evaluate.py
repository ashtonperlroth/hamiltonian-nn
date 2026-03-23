"""Evaluate trained HNN vs baseline on long-term trajectory prediction.

Generates three key plots:
1. Phase space trajectories: true vs HNN vs baseline
2. Energy conservation: H(t) over long rollouts
3. Learned Hamiltonian contours vs true contours (1D systems only)

Usage:
    python -m src.evaluate --system spring
    python -m src.evaluate --system pendulum
    python -m src.evaluate --system twobody
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import torch

from src.baseline import BaselineMLP
from src.hnn import HNN
from src.systems import SYSTEMS


def _eval_model(model, x_tensor):
    """Evaluate model derivatives, handling HNN's autograd requirement."""
    with torch.enable_grad():
        return model(x_tensor).detach().squeeze(0)


def rollout(model, x0, dt, n_steps):
    """Integrate a trajectory using RK4 with the learned model.

    Args:
        model: HNN or BaselineMLP that predicts (dq/dt, dp/dt).
        x0: (state_dim,) initial state tensor.
        dt: time step.
        n_steps: number of integration steps.
    Returns:
        trajectory: (n_steps + 1, state_dim) numpy array.
    """
    model.eval()
    traj = [x0.detach().numpy().copy()]
    x = x0.clone().detach().float()

    for _ in range(n_steps):
        k1 = _eval_model(model, x.unsqueeze(0))
        k2 = _eval_model(model, (x + 0.5 * dt * k1).unsqueeze(0))
        k3 = _eval_model(model, (x + 0.5 * dt * k2).unsqueeze(0))
        k4 = _eval_model(model, (x + dt * k3).unsqueeze(0))

        x = x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        traj.append(x.detach().numpy().copy())

    return np.array(traj)


def compute_energy(traj, system_name):
    """Compute true Hamiltonian energy along a trajectory."""
    sys = SYSTEMS[system_name]
    dof = sys["dof"]
    q = traj[:, :dof]
    p = traj[:, dof:]
    return sys["hamiltonian"](q, p, **sys["params"])


def true_rollout(system_name, x0, dt, n_steps):
    """Integrate a trajectory using scipy (ground truth)."""
    from scipy.integrate import solve_ivp

    sys = SYSTEMS[system_name]
    t_span = (0, dt * n_steps)
    t_eval = np.linspace(0, dt * n_steps, n_steps + 1)
    sol = solve_ivp(
        lambda t, y: sys["dynamics"](t, y, **sys["params"]),
        t_span,
        x0,
        t_eval=t_eval,
        method="DOP853",
        rtol=1e-10,
        atol=1e-10,
    )
    return sol.y.T


def plot_phase_space(true_traj, hnn_traj, bl_traj, system_name, save_dir="figures"):
    """Plot phase space trajectories for 1D systems, or x-y for 2D."""
    os.makedirs(save_dir, exist_ok=True)
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))

    dof = SYSTEMS[system_name]["dof"]
    if dof == 1:
        ax.plot(true_traj[:, 0], true_traj[:, 1], "k-", lw=2, label="Ground truth")
        ax.plot(hnn_traj[:, 0], hnn_traj[:, 1], "b--", lw=1.5, label="HNN")
        ax.plot(bl_traj[:, 0], bl_traj[:, 1], "r--", lw=1.5, label="Baseline")
        ax.set_xlabel("q (position)")
        ax.set_ylabel("p (momentum)")
    else:
        ax.plot(true_traj[:, 0], true_traj[:, 1], "k-", lw=2, label="Ground truth")
        ax.plot(hnn_traj[:, 0], hnn_traj[:, 1], "b--", lw=1.5, label="HNN")
        ax.plot(bl_traj[:, 0], bl_traj[:, 1], "r--", lw=1.5, label="Baseline")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_aspect("equal")

    ax.legend(fontsize=12)
    ax.set_title(f"Phase Space — {system_name}", fontsize=14)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, f"phase_space_{system_name}.png"), dpi=150)
    plt.close(fig)
    print(f"  Saved phase_space_{system_name}.png")


def plot_energy(true_traj, hnn_traj, bl_traj, system_name, dt, save_dir="figures"):
    """Plot energy conservation over time."""
    os.makedirs(save_dir, exist_ok=True)

    E_true = compute_energy(true_traj, system_name)
    E_hnn = compute_energy(hnn_traj, system_name)
    E_bl = compute_energy(bl_traj, system_name)

    t = np.arange(len(E_true)) * dt

    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    ax.plot(t, E_true, "k-", lw=2, label="Ground truth")
    ax.plot(t, E_hnn, "b-", lw=1.5, label="HNN")
    ax.plot(t, E_bl, "r-", lw=1.5, label="Baseline")
    ax.set_xlabel("Time", fontsize=12)
    ax.set_ylabel("Total Energy H", fontsize=12)
    ax.set_title(f"Energy Conservation — {system_name}", fontsize=14)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, f"energy_{system_name}.png"), dpi=150)
    plt.close(fig)
    print(f"  Saved energy_{system_name}.png")

    # Print energy drift stats
    e0 = E_true[0]
    hnn_drift = np.abs(E_hnn - e0).max()
    bl_drift = np.abs(E_bl - e0).max()
    print(f"  Max energy drift — HNN: {hnn_drift:.6f}, Baseline: {bl_drift:.6f}")


def plot_hamiltonian_contours(hnn_model, system_name, save_dir="figures"):
    """Compare learned H contours with true H contours (1D systems only)."""
    if SYSTEMS[system_name]["dof"] != 1:
        return

    os.makedirs(save_dir, exist_ok=True)
    sys = SYSTEMS[system_name]

    # Create grid
    if system_name == "spring":
        q_range = np.linspace(-2.5, 2.5, 100)
        p_range = np.linspace(-2.5, 2.5, 100)
    else:
        q_range = np.linspace(-3.5, 3.5, 100)
        p_range = np.linspace(-10, 10, 100)

    Q, P = np.meshgrid(q_range, p_range)
    grid = np.column_stack([Q.ravel(), P.ravel()])

    # True Hamiltonian
    H_true = sys["hamiltonian"](grid[:, 0:1], grid[:, 1:2], **sys["params"])
    H_true = H_true.reshape(Q.shape)

    # Learned Hamiltonian
    hnn_model.eval()
    with torch.no_grad():
        grid_tensor = torch.tensor(grid, dtype=torch.float32)
        H_learned = hnn_model.hamiltonian(grid_tensor).numpy().reshape(Q.shape)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    levels = 20
    axes[0].contour(Q, P, H_true, levels=levels, cmap="viridis")
    axes[0].set_title("True Hamiltonian", fontsize=14)
    axes[0].set_xlabel("q")
    axes[0].set_ylabel("p")

    axes[1].contour(Q, P, H_learned, levels=levels, cmap="viridis")
    axes[1].set_title("Learned Hamiltonian (HNN)", fontsize=14)
    axes[1].set_xlabel("q")
    axes[1].set_ylabel("p")

    for ax in axes:
        ax.set_aspect("equal" if system_name == "spring" else "auto")
        ax.grid(True, alpha=0.3)

    fig.suptitle(f"Hamiltonian Contours — {system_name}", fontsize=16)
    fig.tight_layout()
    fig.savefig(
        os.path.join(save_dir, f"hamiltonian_contours_{system_name}.png"), dpi=150
    )
    plt.close(fig)
    print(f"  Saved hamiltonian_contours_{system_name}.png")


def evaluate_system(system_name, model_dir="models", n_steps=3000, dt=0.05):
    """Load trained models and run full evaluation."""
    print(f"\n{'='*60}")
    print(f"Evaluating: {system_name}")
    print(f"{'='*60}")

    sys_info = SYSTEMS[system_name]
    state_dim = sys_info["state_dim"]

    # Load models
    hnn = HNN(state_dim)
    hnn.load_state_dict(
        torch.load(os.path.join(model_dir, f"hnn_{system_name}.pt"), weights_only=True)
    )

    baseline = BaselineMLP(state_dim)
    baseline.load_state_dict(
        torch.load(
            os.path.join(model_dir, f"baseline_{system_name}.pt"), weights_only=True
        )
    )

    # Pick initial condition
    rng = np.random.default_rng(123)
    x0_np = sys_info["initial_conditions"](1, rng=rng)[0]
    x0 = torch.tensor(x0_np, dtype=torch.float32)

    print(f"  Initial state: {x0_np}")
    print(f"  Rolling out {n_steps} steps (dt={dt}, T={n_steps * dt:.1f})")

    # Rollouts
    true_traj = true_rollout(system_name, x0_np, dt, n_steps)
    hnn_traj = rollout(hnn, x0, dt, n_steps)
    bl_traj = rollout(baseline, x0, dt, n_steps)

    # Plots
    plot_phase_space(true_traj, hnn_traj, bl_traj, system_name)
    plot_energy(true_traj, hnn_traj, bl_traj, system_name, dt)
    plot_hamiltonian_contours(hnn, system_name)

    return {
        "true_traj": true_traj,
        "hnn_traj": hnn_traj,
        "baseline_traj": bl_traj,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate HNN vs baseline")
    parser.add_argument(
        "--system",
        type=str,
        default="spring",
        choices=list(SYSTEMS.keys()),
    )
    parser.add_argument("--n_steps", type=int, default=3000)
    parser.add_argument("--dt", type=float, default=0.05)
    args = parser.parse_args()

    evaluate_system(args.system, n_steps=args.n_steps, dt=args.dt)


if __name__ == "__main__":
    main()
