"""Outcome-space partitions and evaluation metrics.

The paper measures exploration with *reachability*: the fraction of
behaviour regions of a partition of outcome space that contain at least one
observed outcome,

.. math::  \\mathrm{Reach}_t = |\\{\\varphi(y_i): i \\le t\\}| / |\\mathcal{B}|.

A partition maps outcomes ``Y`` (``N x m``) to a boolean membership matrix
(``N x M``) over ``M`` regions. The same partitions define the behaviour
regions that guide UG-BEACON (:mod:`beacon.algorithms`).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable, Optional, Sequence, Union

import numpy as np
import torch

Number = Union[int, float]


def _as_2d(Y: torch.Tensor) -> torch.Tensor:
    Y = torch.as_tensor(Y)
    return Y.unsqueeze(-1) if Y.dim() == 1 else Y


class OutcomePartition(ABC):
    """A collection of ``n_regions`` behaviour regions in outcome space."""

    n_regions: int

    @abstractmethod
    def membership(self, Y: torch.Tensor) -> torch.Tensor:
        """Boolean ``(N, n_regions)`` matrix: does outcome ``i`` lie in region ``j``?"""

    def visited(self, Y: torch.Tensor) -> torch.Tensor:
        """Boolean ``(n_regions,)`` mask of regions containing any outcome in ``Y``."""
        Y = _as_2d(Y)
        if Y.shape[0] == 0:
            return torch.zeros(self.n_regions, dtype=torch.bool)
        return self.membership(Y).any(dim=0).cpu()

    def counts(self, Y: torch.Tensor) -> torch.Tensor:
        """Number of outcomes per region, shape ``(n_regions,)``."""
        return self.membership(_as_2d(Y)).sum(dim=0).cpu()


class _IndexedPartition(OutcomePartition):
    """Partition in which each outcome falls in at most one region."""

    @abstractmethod
    def region_index(self, Y: torch.Tensor) -> torch.Tensor:
        """Region index per outcome, ``-1`` when the outcome lies in no region."""

    def membership(self, Y: torch.Tensor) -> torch.Tensor:
        idx = self.region_index(_as_2d(Y))
        member = torch.zeros(idx.shape[0], self.n_regions, dtype=torch.bool, device=idx.device)
        inside = idx >= 0
        member[inside.nonzero(as_tuple=True)[0], idx[inside]] = True
        return member

    def visited(self, Y: torch.Tensor) -> torch.Tensor:
        Y = _as_2d(Y)
        mask = torch.zeros(self.n_regions, dtype=torch.bool)
        idx = self.region_index(Y).cpu()
        mask[idx[idx >= 0]] = True
        return mask


class HistogramPartition(_IndexedPartition):
    """``n_bins`` equal-width bins over ``[lower, upper]`` of a scalar outcome.

    Bin assignment follows :func:`numpy.histogram` exactly (half-open bins,
    last bin closed; outcomes outside the range belong to no bin), which is
    what the paper's single-outcome reachability used.
    """

    def __init__(self, lower: Number, upper: Number, n_bins: int = 25):
        if not upper > lower:
            raise ValueError("upper must be larger than lower")
        self.lower, self.upper, self.n_bins = float(lower), float(upper), int(n_bins)
        self.n_regions = self.n_bins
        self.edges = torch.from_numpy(np.linspace(self.lower, self.upper, self.n_bins + 1))

    def region_index(self, Y: torch.Tensor) -> torch.Tensor:
        Y = _as_2d(Y)
        if Y.shape[-1] != 1:
            raise ValueError("HistogramPartition expects a single outcome column.")
        y = Y[:, 0].detach().to(dtype=torch.double, device="cpu")
        idx = torch.searchsorted(self.edges, y, right=True) - 1
        idx = torch.where(y == self.edges[-1], torch.full_like(idx, self.n_bins - 1), idx)
        idx[(idx < 0) | (idx >= self.n_bins) | torch.isnan(y)] = -1
        return idx


class GridPartition(_IndexedPartition):
    """Uniform grid with ``n_bins`` cells per outcome dimension.

    Cell ``floor((y - lower) / cell_size)`` is clamped to ``n_bins - 1`` so an
    outcome exactly at ``upper`` falls in the last cell (instead of a spurious,
    zero-width extra cell). Outcomes outside ``[lower, upper]`` belong to no cell.
    """

    def __init__(self, lower: Sequence[Number], upper: Sequence[Number], n_bins: Union[int, Sequence[int]] = 10):
        self.lower = torch.as_tensor(lower, dtype=torch.double).reshape(-1)
        self.upper = torch.as_tensor(upper, dtype=torch.double).reshape(-1)
        m = self.lower.numel()
        bins = [int(n_bins)] * m if isinstance(n_bins, (int, np.integer)) else [int(b) for b in n_bins]
        if len(bins) != m or self.upper.numel() != m:
            raise ValueError("lower, upper and n_bins must have one entry per outcome")
        self.n_bins = torch.tensor(bins)
        self.n_regions = int(np.prod(bins))
        self._strides = torch.tensor([int(np.prod(bins[i + 1 :])) for i in range(m)])

    def region_index(self, Y: torch.Tensor) -> torch.Tensor:
        Y = _as_2d(Y).detach().to(dtype=torch.double, device="cpu")
        cell_size = (self.upper - self.lower) / self.n_bins
        cells = torch.floor((Y - self.lower) / cell_size).long()
        cells = torch.minimum(cells, self.n_bins - 1)
        inside = ((Y >= self.lower) & (Y <= self.upper)).all(dim=-1)
        flat = (cells * self._strides).sum(-1)
        return torch.where(inside, flat, torch.full_like(flat, -1))


class BoxPartition(OutcomePartition):
    """Arbitrary (closed, possibly non-uniform) axis-aligned boxes.

    Used for the oil-sorbent study, whose outcome space is partitioned more
    finely in high-utility regions. Optional ``clip_lower``/``clip_upper``
    clamp outcomes before testing membership (used by UG-BEACON so that
    extrapolated Thompson samples are assigned to the nearest boundary box).
    """

    def __init__(
        self,
        lower_bounds: torch.Tensor,
        upper_bounds: torch.Tensor,
        clip_lower: Optional[Sequence[Number]] = None,
        clip_upper: Optional[Sequence[Number]] = None,
    ):
        self.lower_bounds = torch.as_tensor(lower_bounds, dtype=torch.double)
        self.upper_bounds = torch.as_tensor(upper_bounds, dtype=torch.double)
        if self.lower_bounds.shape != self.upper_bounds.shape or self.lower_bounds.dim() != 2:
            raise ValueError("lower_bounds and upper_bounds must both be (n_regions, m)")
        self.n_regions = self.lower_bounds.shape[0]
        self.clip_lower = None if clip_lower is None else torch.as_tensor(clip_lower, dtype=torch.double)
        self.clip_upper = None if clip_upper is None else torch.as_tensor(clip_upper, dtype=torch.double)

    def membership(self, Y: torch.Tensor) -> torch.Tensor:
        Y = _as_2d(Y).detach().to(dtype=torch.double, device="cpu")
        if self.clip_lower is not None or self.clip_upper is not None:
            Y = Y.clamp(min=self.clip_lower, max=self.clip_upper)
        Y = Y.unsqueeze(1)
        return ((Y >= self.lower_bounds) & (Y <= self.upper_bounds)).all(dim=-1)

    @classmethod
    def from_grid(cls, lower: Sequence[Number], upper: Sequence[Number], partitions: Sequence[int]) -> "BoxPartition":
        """Regular grid of boxes over ``[lower, upper]`` with ``partitions[i]`` cells per axis."""
        edges = [torch.linspace(float(lo), float(hi), p + 1, dtype=torch.double) for lo, hi, p in zip(lower, upper, partitions)]
        mesh = torch.meshgrid(*[torch.arange(p) for p in partitions], indexing="ij")
        idx = torch.stack([g.reshape(-1) for g in mesh], dim=-1)
        lb = torch.stack([edges[d][idx[:, d]] for d in range(len(partitions))], dim=-1)
        ub = torch.stack([edges[d][idx[:, d] + 1] for d in range(len(partitions))], dim=-1)
        return cls(lb, ub)

    @classmethod
    def concat(cls, parts: Sequence["BoxPartition"], **kwargs) -> "BoxPartition":
        return cls(
            torch.cat([p.lower_bounds for p in parts]), torch.cat([p.upper_bounds for p in parts]), **kwargs
        )


class ClassifierPartition(_IndexedPartition):
    """Regions defined by a classifier's predicted label.

    ``classify(Y)`` must return ``(labels, confidences)``; outcomes whose
    confidence is below ``threshold`` belong to no region. Used for MNIST,
    where a CNN assigns decoded images to the ten digit classes.
    """

    def __init__(self, classify: Callable[[torch.Tensor], tuple], n_classes: int, threshold: float = 0.0):
        self.classify = classify
        self.n_regions = int(n_classes)
        self.threshold = float(threshold)

    def region_index(self, Y: torch.Tensor) -> torch.Tensor:
        labels, confidence = self.classify(_as_2d(Y))
        labels = labels.long().cpu()
        return torch.where(confidence.cpu() >= self.threshold, labels, torch.full_like(labels, -1))


class Reachability:
    """Fraction of partition regions reached by a set of outcomes.

    Args:
        partition: The behaviour partition.
        normalizer: Denominator ``|B|``. ``None`` uses ``partition.n_regions``.
            For finite datasets the paper normalizes by the number of regions
            that are reachable at all; see :meth:`from_dataset`.
    """

    def __init__(self, partition: OutcomePartition, normalizer: Optional[Number] = None):
        self.partition = partition
        self.normalizer = float(partition.n_regions if normalizer is None else normalizer)
        if self.normalizer <= 0:
            raise ValueError("normalizer must be positive")

    @classmethod
    def from_dataset(cls, partition: OutcomePartition, Y_all: torch.Tensor) -> "Reachability":
        """Normalize by the number of regions occupied by the full candidate set."""
        return cls(partition, normalizer=int(partition.visited(Y_all).sum()))

    def __call__(self, Y: torch.Tensor) -> float:
        return float(self.partition.visited(Y).sum()) / self.normalizer


def uniformity(Y: torch.Tensor, partition: OutcomePartition) -> float:
    """``1 - JS(p, u)`` between occupied-region frequencies and a uniform law.

    Reported as a secondary metric by some original scripts (base-2
    Jensen-Shannon distance, computed over occupied regions only).
    """
    from scipy.spatial.distance import jensenshannon

    counts = partition.counts(Y).double().numpy()
    counts = counts[counts > 0]
    if counts.size == 0:
        return float("nan")
    p = counts / counts.sum()
    return float(1.0 - jensenshannon(p, np.full_like(p, 1.0 / p.size), base=2))


class BestValue:
    """Best value of a scalar score of the outcomes (e.g. maze reward)."""

    def __init__(self, score: Callable[[torch.Tensor], torch.Tensor]):
        self.score = score

    def __call__(self, Y: torch.Tensor) -> float:
        return float(self.score(_as_2d(Y)).max())
