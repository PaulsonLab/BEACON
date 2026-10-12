"""Synthetic continuous benchmarks (paper Sec. 5.1).

Settings were verified against the tracked results in
``results/continuous-single-outcome``: the saved initial reachabilities are
reproduced exactly with ``n_init = 10`` uniform random points (seeds 0-19),
inputs scaled from ``[0, 1]^d`` to ``[-5, 5]^d`` and 25 bins over the outcome
ranges below. (The paper text states 20 initial points; the released results
used 10.)
"""

from __future__ import annotations

import torch
from botorch.test_functions import Ackley, Rosenbrock, StyblinskiTang

from beacon.metrics import GridPartition, HistogramPartition, Reachability
from beacon.problems.base import ContinuousProblem

SYNTHETIC_DEFAULTS = dict(
    n_init=10,
    iterations=200,
    replicates=20,
    k=10,
    kernel="rbf",
    likelihood="gamma",
    archive="observed",
)

# Outcome ranges [obj_lb, obj_ub] over [-5, 5]^d used by the original scripts.
_ROSENBROCK_UB = {4: 270108.0, 8: 630252.63, 12: 990396.990397}


def _outcome_range(name: str, dim: int):
    if name == "ackley":
        return 0.0, 14.3027
    if name == "rosenbrock":
        # max at x = (-5, ..., -5): (d - 1) * (100 * 30^2 + 6^2)
        return 0.0, _ROSENBROCK_UB.get(dim, (dim - 1) * 90036.0)
    if name == "styblinski-tang":
        return -39.16599 * dim, 125.0 * dim
    raise ValueError(name)


_FUNCTIONS = {"ackley": Ackley, "rosenbrock": Rosenbrock, "styblinski-tang": StyblinskiTang}


def make_synthetic(name: str, dim: int = 4, n_bins: int = 25, noise_std: float = 0.0, **_) -> ContinuousProblem:
    """Ackley / Rosenbrock / Styblinski-Tang on ``[-5, 5]^dim`` with a scalar outcome."""
    dim = int(dim)
    function = _FUNCTIONS[name](dim=dim)

    def objective(X: torch.Tensor) -> torch.Tensor:
        return function.evaluate_true(X).unsqueeze(-1)

    lo, hi = _outcome_range(name, dim)
    partition = HistogramPartition(lo, hi, n_bins)
    return ContinuousProblem(
        name=f"{name}-{dim}d",
        objective=objective,
        input_dim=dim,
        output_dim=1,
        lower=-5.0,
        upper=5.0,
        metric=Reachability(partition),
        partition=partition,
        noise_std=noise_std,
        defaults=dict(SYNTHETIC_DEFAULTS),
        description=f"{dim}D {name.replace('-', ' ').title()} function, reachability over {n_bins} bins of [{lo:g}, {hi:g}].",
    )


def make_noisy_ackley(dim: int = 4, noise_std: float = 0.5, n_bins: int = 50, **_) -> ContinuousProblem:
    """Ackley with additive Gaussian observation noise (paper noise study).

    Reachability is computed from the *noise-free* outcomes of the selected
    inputs. BEACON models the noise (``likelihood='plain'``) and measures
    novelty against the GP posterior mean (``archive='posterior_mean'``). The
    "noiseless" baseline of the paper corresponds to
    ``likelihood='noiseless', archive='observed'``.
    """
    problem = make_synthetic("ackley", dim=dim, n_bins=n_bins, noise_std=noise_std)
    problem.name = f"noisy-ackley-{int(dim)}d"
    problem.defaults.update(n_init=50, iterations=300, likelihood="plain", archive="posterior_mean")
    problem.description = f"{int(dim)}D Ackley with N(0, {noise_std}^2) observation noise, {n_bins} bins."
    return problem


def multi_output_plus(x: torch.Tensor) -> torch.Tensor:
    """Two-outcome synthetic function of 6 inputs from the paper ("Multi-Output Plus")."""
    y1 = (
        torch.sin(x[:, 0]) * torch.cos(x[:, 1])
        + x[:, 2] * torch.exp(-x[:, 0] ** 2) * torch.cos(x[:, 0] + x[:, 1])
        + 0.01 * torch.sin(x[:, 3] + x[:, 4] + x[:, 5])
    )
    y2 = (
        torch.sin(x[:, 3]) * torch.cos(x[:, 4])
        + x[:, 5] * torch.exp(-x[:, 3] ** 2) * torch.sin(x[:, 3] + x[:, 4])
        + 0.01 * torch.cos(x[:, 0] + x[:, 1] + x[:, 2])
    )
    return torch.stack((y1, y2), dim=-1)


def make_multi_output_plus(n_bins: int = 10, **_) -> ContinuousProblem:
    """6D, 2-outcome synthetic problem; reachability on a 10 x 10 grid of [-5.1, 5.1]^2."""
    partition = GridPartition([-5.1, -5.1], [5.1, 5.1], n_bins)
    return ContinuousProblem(
        name="multi-output-plus",
        objective=multi_output_plus,
        input_dim=6,
        output_dim=2,
        lower=-5.0,
        upper=5.0,
        metric=Reachability(partition),
        partition=partition,
        defaults=dict(SYNTHETIC_DEFAULTS, iterations=500),
        description=f"Multi-Output Plus (6 inputs, 2 outcomes), {n_bins}x{n_bins} reachability grid.",
    )
