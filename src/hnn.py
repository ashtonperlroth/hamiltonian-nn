"""Hamiltonian Neural Network.

The network learns a scalar function H(q, p) and uses torch.autograd to
compute the symplectic gradient — Hamilton's equations:
    dq/dt =  ∂H/∂p
    dp/dt = -∂H/∂q

This architectural constraint guarantees energy conservation.
"""

import torch
import torch.nn as nn


class HNN(nn.Module):
    """Hamiltonian Neural Network.

    Args:
        input_dim: dimension of the full state (q, p). Must be even —
            first half is q, second half is p.
        hidden_dim: width of hidden layers.
        n_layers: number of hidden layers.
    """

    def __init__(self, input_dim, hidden_dim=200, n_layers=3):
        super().__init__()
        assert input_dim % 2 == 0, "input_dim must be even (q, p pairs)"
        self.dof = input_dim // 2

        layers = [nn.Linear(input_dim, hidden_dim), nn.Tanh()]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.Tanh()]
        layers.append(nn.Linear(hidden_dim, 1))
        self.net = nn.Sequential(*layers)

    def hamiltonian(self, x):
        """Compute the learned Hamiltonian H(q, p).

        Args:
            x: (batch, input_dim) tensor of (q, p) states.
        Returns:
            H: (batch, 1) tensor of scalar energy values.
        """
        return self.net(x)

    def forward(self, x):
        """Compute time derivatives via Hamilton's equations.

        Args:
            x: (batch, input_dim) tensor of (q, p) states.
        Returns:
            dxdt: (batch, input_dim) tensor of (dq/dt, dp/dt).
        """
        x = x.requires_grad_(True)
        H = self.hamiltonian(x)
        dH = torch.autograd.grad(H.sum(), x, create_graph=True)[0]

        dH_dq = dH[:, : self.dof]
        dH_dp = dH[:, self.dof :]

        # Hamilton's equations
        dqdt = dH_dp
        dpdt = -dH_dq
        return torch.cat([dqdt, dpdt], dim=1)
