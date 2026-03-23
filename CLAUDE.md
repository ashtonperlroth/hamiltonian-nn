# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Hamiltonian Neural Network (HNN) — learns an unknown energy function H(q,p) from trajectory data, then uses Hamilton's equations (dq/dt = ∂H/∂p, dp/dt = -∂H/∂q) via autograd to compute dynamics. This architectural constraint automatically conserves energy. **This is NOT a PINN** — PINNs embed known equations in the loss; HNNs learn unknown physics from data.

Based on: Greydanus, Dzamba & Sosanya, "Hamiltonian Neural Networks" (arXiv:1906.01563, NeurIPS 2019).

## Architecture

- **`src/systems.py`** — Generates trajectory data (q, p, dq/dt, dp/dt) for three systems: mass-spring (1D harmonic oscillator), ideal pendulum, two-body gravitational problem. Uses scipy to integrate Hamilton's equations.
- **`src/hnn.py`** — HNN: MLP maps (q,p) → scalar H, then torch.autograd computes the symplectic gradient (Hamilton's equations). The key function is the forward pass that returns predicted time derivatives.
- **`src/baseline.py`** — Vanilla MLP baseline: same architecture but directly predicts (dq/dt, dp/dt) without Hamiltonian structure.
- **`src/train.py`** — Training loop for both models. Loss = MSE between predicted and true derivatives.
- **`src/evaluate.py`** — Long-rollout comparison: phase space trajectories, energy conservation over time, learned vs true Hamiltonian contours.
- **`tests/test_hamiltonians.py`** — Tests for data generation, gradient computation, and energy conservation.
- **`notebooks/results.ipynb`** — Visualization notebook.

## Development Setup

- **Framework:** PyTorch (CPU-only for these toy problems)
- **Formatter:** Black (auto-runs on save via hook)
- **Linter:** Ruff (auto-runs with `--fix` on save via hook)
- **Testing:** pytest

## Commands

- Run all tests: `python -m pytest`
- Run a single test: `python -m pytest tests/test_hamiltonians.py::test_name`
- Train a model: `python -m src.train --system spring` (also: `pendulum`, `twobody`)
- Evaluate: `python -m src.evaluate --system spring`
- Format: `black .`
- Lint: `ruff check --fix .`
- Install deps: `pip install -r requirements.txt`

## Key Design Decisions

- All systems use canonical coordinates (q, p) — generalized position and momentum.
- The HNN outputs a **scalar** H and uses `torch.autograd.grad` to get derivatives. This is the core architectural constraint that enforces energy conservation.
- Trajectory rollouts use a simple leapfrog/symplectic integrator for fair comparison.
- Training data includes small Gaussian noise to simulate measurement error.
