"""Vanilla MLP baseline for predicting dynamics.

Directly maps (q, p) → (dq/dt, dp/dt) without any Hamiltonian structure.
Same capacity as the HNN for fair comparison.
"""

import torch.nn as nn


class BaselineMLP(nn.Module):
    """Standard MLP that directly predicts time derivatives.

    Args:
        input_dim: dimension of the full state (q, p).
        hidden_dim: width of hidden layers.
        n_layers: number of hidden layers.
    """

    def __init__(self, input_dim, hidden_dim=200, n_layers=3):
        super().__init__()
        layers = [nn.Linear(input_dim, hidden_dim), nn.Tanh()]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.Tanh()]
        layers.append(nn.Linear(hidden_dim, input_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        """Predict time derivatives directly.

        Args:
            x: (batch, input_dim) tensor of (q, p) states.
        Returns:
            dxdt: (batch, input_dim) tensor of (dq/dt, dp/dt).
        """
        return self.net(x)
