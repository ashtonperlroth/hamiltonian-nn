# Hamiltonian Neural Networks

A neural network that learns to predict dynamics by learning an unknown energy function from data, then using Hamilton's equations to compute time evolution — automatically conserving energy by construction.

## The Core Idea

Instead of learning dynamics directly (q, p) → (dq/dt, dp/dt), the network learns a scalar Hamiltonian H(q, p) and computes derivatives via the symplectic gradient:

```
dq/dt =  ∂H/∂p
dp/dt = -∂H/∂q
```

These are Hamilton's equations of motion. Because the dynamics live on level sets of H, **energy is conserved automatically** — the network doesn't need to learn conservation laws; the architecture makes violating them impossible.

```
         ┌─────────────┐
(q, p) → │  MLP → H(q,p) │ → autograd → (∂H/∂p, -∂H/∂q) → (dq/dt, dp/dt)
         └─────────────┘
              scalar
```

**This is NOT a physics-informed neural network (PINN).** PINNs embed a *known* equation into the loss function. HNNs learn an *unknown* energy function from trajectory data and use Hamiltonian structure as an architectural constraint. It's learning physics, not solving it.

Based on: Greydanus, Dzamba & Sosanya, ["Hamiltonian Neural Networks"](https://arxiv.org/abs/1906.01563) (NeurIPS 2019).

## Systems

Three physical systems of increasing difficulty:

| System | Hamiltonian | DOF | Description |
|--------|------------|-----|-------------|
| Mass-spring | H = p²/2m + kq²/2 | 1D | Linear harmonic oscillator |
| Pendulum | H = p²/2ml² - mgl·cos(q) | 1D | Nonlinear, conserves energy |
| Two-body | H = \|p\|²/2m - GMm/\|q\| | 2D | Keplerian orbits |

## Installation

```bash
git clone https://github.com/your-username/hamiltonian-nn.git
cd hamiltonian-nn
pip install -r requirements.txt
```

Runs on CPU — no GPU required for these toy problems. A GPU would help if scaling to higher-dimensional systems or larger datasets.

## Usage

**Train models:**
```bash
python -m src.train --system spring      # ~30 seconds
python -m src.train --system pendulum    # ~30 seconds
python -m src.train --system twobody     # ~30 seconds
```

**Evaluate and generate plots:**
```bash
python -m src.evaluate --system spring
python -m src.evaluate --system pendulum
python -m src.evaluate --system twobody
```

Plots are saved to `figures/`.

**Run the notebook:**
```bash
jupyter notebook notebooks/results.ipynb
```

**Run tests:**
```bash
python -m pytest
```

## Key Results

After training both models on the same data, roll out long trajectories (150 time units, 100+ oscillation periods):

- **Phase space**: HNN orbits close; baseline spirals outward (energy drift)
- **Energy conservation**: HNN maintains near-constant H(t); baseline drifts monotonically
- **Learned Hamiltonian**: HNN contours match true Hamiltonian contours (up to a constant offset)

The critical insight: the HNN and baseline have **similar training loss** (both learn the local derivatives well), but the HNN vastly outperforms on long-term prediction because its architecture enforces the correct geometric structure.

## Project Structure

```
src/
├── systems.py    # Trajectory data generation (scipy integration)
├── hnn.py        # Hamiltonian Neural Network (MLP → scalar H → autograd)
├── baseline.py   # Vanilla MLP baseline (direct derivative prediction)
├── train.py      # Training loop (Adam + cosine annealing)
└── evaluate.py   # Long-rollout comparison and plotting
tests/
└── test_hamiltonians.py
notebooks/
└── results.ipynb
```
