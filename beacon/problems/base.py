"""Problem interface consumed by :mod:`beacon.runner`.

A *continuous* problem exposes an objective on the unit cube ``[0, 1]^d``
(inputs are rescaled internally to the physical domain, as in the original
scripts). A *discrete* problem exposes a finite candidate set with known
outcomes (a dataset), from which BEACON may only select unseen candidates.

Each problem also carries:

* ``metric``: callable mapping noise-free outcomes ``(N, m)`` to a float
  (reachability for every study except the maze, which reports the best
  reward),
* ``partition``: behaviour regions used by UG-BEACON,
* ``defaults``: the paper's experimental settings (``n_init``,
  ``iterations``, ``replicates``) and BEACON options (``k``, kernel, ...).
"""

from __future__ import annotations

from typing import Callable, Dict, Optional, Sequence, Tuple

import numpy as np
import torch

from beacon.device import DTYPE
from beacon.metrics import OutcomePartition

Metric = Callable[[torch.Tensor], float]


class Problem:
    """Common attributes of continuous and discrete problems."""

    is_discrete: bool = False

    def __init__(
        self,
        name: str,
        input_dim: int,
        output_dim: int,
        metric: Metric,
        partition: Optional[OutcomePartition] = None,
        defaults: Optional[Dict] = None,
        description: str = "",
        output_shape: Optional[Sequence[int]] = None,
        metric_name: str = "reachability",
    ):
        self.name = name
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.metric = metric
        self.partition = partition
        self.defaults = dict(defaults or {})
        self.description = description
        self.output_shape = tuple(output_shape) if output_shape is not None else None
        self.metric_name = metric_name

    @property
    def guidance_partition(self) -> Optional[OutcomePartition]:
        """Behaviour regions UG-BEACON avoids once visited (defaults to ``partition``)."""
        return getattr(self, "_guidance_partition", None) or self.partition

    @guidance_partition.setter
    def guidance_partition(self, value: Optional[OutcomePartition]) -> None:
        self._guidance_partition = value

    def __repr__(self) -> str:
        kind = "discrete" if self.is_discrete else "continuous"
        return f"{type(self).__name__}(name={self.name!r}, {kind}, d={self.input_dim}, m={self.output_dim})"


class ContinuousProblem(Problem):
    """Black-box function on a box domain, optimized on ``[0, 1]^d``.

    Args:
        objective: maps physical inputs ``(n, d)`` to noise-free outcomes ``(n, m)``.
        lower, upper: physical box bounds (scalars or length-``d`` sequences).
        noise_std: std of additive Gaussian observation noise.
    """

    is_discrete = False

    def __init__(
        self,
        name: str,
        objective: Callable[[torch.Tensor], torch.Tensor],
        input_dim: int,
        output_dim: int,
        lower,
        upper,
        metric: Metric,
        noise_std: float = 0.0,
        **kwargs,
    ):
        super().__init__(name, input_dim, output_dim, metric, **kwargs)
        self.objective = objective
        self.lower = torch.as_tensor(lower, dtype=DTYPE).expand(self.input_dim).clone()
        self.upper = torch.as_tensor(upper, dtype=DTYPE).expand(self.input_dim).clone()
        self.noise_std = float(noise_std)

    def unit_bounds(self, device=None) -> torch.Tensor:
        """``2 x d`` bounds of the unit cube the acquisition is optimized over."""
        return torch.stack([torch.zeros(self.input_dim), torch.ones(self.input_dim)]).to(dtype=DTYPE, device=device)

    def to_physical(self, X_unit: torch.Tensor) -> torch.Tensor:
        lower = self.lower.to(X_unit)
        upper = self.upper.to(X_unit)
        return lower + (upper - lower) * X_unit

    def evaluate(self, X_unit: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return ``(observed, noise_free)`` outcomes, each ``(n, m)``."""
        y_true = self.objective(self.to_physical(X_unit))
        y_true = y_true.reshape(X_unit.shape[0], self.output_dim).to(X_unit)
        if self.noise_std > 0:
            return y_true + self.noise_std * torch.randn_like(y_true), y_true
        return y_true, y_true

    def initial_design(self, n_init: int, seed: int) -> torch.Tensor:
        """Uniform random design on ``[0, 1]^d`` (``np.random.seed(seed); rand(n, d)``)."""
        rng = np.random.RandomState(seed)
        return torch.as_tensor(rng.rand(n_init, self.input_dim), dtype=DTYPE)

    def initial_data(self, n_init: int, seed: int, device=None):
        """Initial inputs and their ``(observed, noise_free)`` outcomes."""
        X = self.initial_design(n_init, seed).to(device)
        Y_obs, Y_true = self.evaluate(X)
        return X, Y_obs, Y_true


class DiscreteProblem(Problem):
    """Finite candidate set ``X`` (``N x d``) with known outcomes ``Y`` (``N x m``).

    ``Y_true`` (defaults to ``Y``) are the outcomes used by the metric.
    """

    is_discrete = True

    def __init__(self, name: str, X: torch.Tensor, Y: torch.Tensor, metric: Metric, Y_true=None, **kwargs):
        X = torch.as_tensor(X).to(DTYPE)
        Y = torch.as_tensor(Y).to(DTYPE)
        if Y.dim() == 1:
            Y = Y.unsqueeze(-1)
        if X.shape[0] != Y.shape[0]:
            raise ValueError("X and Y must have the same number of rows")
        super().__init__(name, X.shape[-1], Y.shape[-1], metric, **kwargs)
        self.X = X
        self.Y = Y
        self.Y_true = Y if Y_true is None else torch.as_tensor(Y_true).to(DTYPE).reshape(Y.shape)

    @property
    def n_candidates(self) -> int:
        return self.X.shape[0]

    def initial_indices(self, n_init: int, seed: int) -> np.ndarray:
        """``np.random.seed(seed); choice(N, n_init, replace=False)`` as in the paper."""
        rng = np.random.RandomState(seed)
        return rng.choice(np.arange(self.n_candidates), size=n_init, replace=False)


def minmax_normalize(X: torch.Tensor) -> torch.Tensor:
    """Scale each column to ``[0, 1]``; constant columns map to 0."""
    lo = X.min(dim=0).values
    span = X.max(dim=0).values - lo
    span = torch.where(span > 0, span, torch.ones_like(span))
    return (X - lo) / span
