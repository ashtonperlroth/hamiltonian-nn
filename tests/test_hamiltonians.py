"""Tests for the Hamiltonian Neural Network project."""

import numpy as np
import pytest
import torch

from src.baseline import BaselineMLP
from src.hnn import HNN
from src.systems import (
    SYSTEMS,
    generate_trajectories,
    spring_dynamics,
    spring_hamiltonian,
)


# ---------------------------------------------------------------------------
# System tests
# ---------------------------------------------------------------------------


class TestSystems:
    """Test trajectory data generation."""

    @pytest.mark.parametrize("system_name", ["spring", "pendulum", "twobody"])
    def test_generate_trajectories_shape(self, system_name):
        data = generate_trajectories(system_name, n_trajectories=3, t_span=(0, 2))
        state_dim = SYSTEMS[system_name]["state_dim"]
        assert data["states"].shape[1] == state_dim
        assert data["derivatives"].shape[1] == state_dim
        assert data["states"].shape[0] == data["derivatives"].shape[0]
        assert data["states"].shape[0] == data["energies"].shape[0]

    @pytest.mark.parametrize("system_name", ["spring", "pendulum", "twobody"])
    def test_energy_conservation_in_data(self, system_name):
        """Verify that the integrator conserves energy (no noise)."""
        data = generate_trajectories(
            system_name, n_trajectories=1, t_span=(0, 5), noise_std=0.0
        )
        # Energy should be nearly constant along each trajectory
        n_steps = len(data["t"])
        energies = data["energies"][:n_steps]  # first trajectory only
        energy_drift = np.abs(energies - energies[0]).max()
        assert energy_drift < 1e-6, f"Energy drift too large: {energy_drift}"

    def test_spring_hamiltonian_values(self):
        q, p = 1.0, 0.0
        H = spring_hamiltonian(q, p, m=1.0, k=1.0)
        assert np.isclose(H, 0.5)  # H = 0 + k*q²/2 = 0.5

        q, p = 0.0, 1.0
        H = spring_hamiltonian(q, p, m=1.0, k=1.0)
        assert np.isclose(H, 0.5)  # H = p²/2 + 0 = 0.5

    def test_spring_dynamics_hamiltons_equations(self):
        """dq/dt = p/m, dp/dt = -kq for mass-spring."""
        q, p = 2.0, 3.0
        dqdt, dpdt = spring_dynamics(0, [q, p], m=1.0, k=1.0)
        assert np.isclose(dqdt, 3.0)  # p/m
        assert np.isclose(dpdt, -2.0)  # -k*q

    def test_noise_injection(self):
        data_clean = generate_trajectories("spring", n_trajectories=5, noise_std=0.0)
        data_noisy = generate_trajectories(
            "spring", n_trajectories=5, noise_std=0.1, rng=42
        )
        # Noisy derivatives should differ from clean
        assert not np.allclose(
            data_clean["derivatives"], data_noisy["derivatives"], atol=1e-3
        )


# ---------------------------------------------------------------------------
# Model tests
# ---------------------------------------------------------------------------


class TestHNN:
    """Test HNN architecture and gradient computation."""

    def test_output_shape(self):
        hnn = HNN(input_dim=2)
        x = torch.randn(16, 2)
        out = hnn(x)
        assert out.shape == (16, 2)

    def test_output_shape_4d(self):
        hnn = HNN(input_dim=4)
        x = torch.randn(16, 4)
        out = hnn(x)
        assert out.shape == (16, 4)

    def test_hamiltonian_is_scalar(self):
        hnn = HNN(input_dim=2)
        x = torch.randn(8, 2)
        H = hnn.hamiltonian(x)
        assert H.shape == (8, 1)

    def test_gradients_flow(self):
        """Verify that gradients propagate through the autograd computation."""
        hnn = HNN(input_dim=2)
        x = torch.randn(4, 2)
        target = torch.randn(4, 2)
        out = hnn(x)
        loss = ((out - target) ** 2).sum()
        loss.backward()
        # The final bias has no gradient when loss = out.sum() because the
        # HNN differentiates H — the bias vanishes. With MSE loss against a
        # target, all parameters except the final bias should get gradients.
        grads = [p.grad for p in hnn.parameters() if p.grad is not None]
        assert len(grads) >= len(list(hnn.parameters())) - 1

    def test_symplectic_structure(self):
        """The HNN should produce derivatives consistent with Hamilton's equations.

        Specifically: if we compute dH/dq and dH/dp numerically, we should get
        dq/dt ≈ dH/dp and dp/dt ≈ -dH/dq.
        """
        hnn = HNN(input_dim=2)
        x = torch.tensor([[1.0, 0.5]], requires_grad=True)

        # Get model predictions
        dxdt = hnn(x)
        dqdt_pred = dxdt[0, 0].item()
        dpdt_pred = dxdt[0, 1].item()

        # Numerical gradients of H
        eps = 1e-4
        x_base = torch.tensor([[1.0, 0.5]])
        with torch.no_grad():
            H_qp = hnn.hamiltonian(x_base).item()
            H_qp_dq = hnn.hamiltonian(x_base + torch.tensor([[eps, 0.0]])).item()
            H_qp_dp = hnn.hamiltonian(x_base + torch.tensor([[0.0, eps]])).item()

        dH_dq_num = (H_qp_dq - H_qp) / eps
        dH_dp_num = (H_qp_dp - H_qp) / eps

        # Hamilton's equations: dq/dt = dH/dp, dp/dt = -dH/dq
        assert abs(dqdt_pred - dH_dp_num) < 1e-2
        assert abs(dpdt_pred - (-dH_dq_num)) < 1e-2


class TestBaseline:
    def test_output_shape(self):
        bl = BaselineMLP(input_dim=2)
        x = torch.randn(16, 2)
        assert bl(x).shape == (16, 2)

    def test_output_shape_4d(self):
        bl = BaselineMLP(input_dim=4)
        x = torch.randn(16, 4)
        assert bl(x).shape == (16, 4)


# ---------------------------------------------------------------------------
# Integration test
# ---------------------------------------------------------------------------


class TestTraining:
    """Quick integration test to verify training doesn't crash."""

    def test_train_spring_few_epochs(self):
        from src.train import make_dataset, train_model

        train_ds, test_ds = make_dataset("spring", n_trajectories=5)
        hnn = HNN(2)
        history = train_model(hnn, train_ds, test_ds, epochs=10, batch_size=64)
        assert len(history["train_loss"]) == 10
        # Loss should decrease
        assert history["train_loss"][-1] < history["train_loss"][0]
