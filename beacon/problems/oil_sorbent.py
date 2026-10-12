"""Oil-sorbent design benchmark (paper Sec. 5.5, UG-BEACON).

Seven process variables are mapped to two outcomes, water-contact-angle-
corrected oil absorption capacity ``q`` and mechanical strength ``sigma``,
through fitted response-surface models. A fixed 5,000-point Latin hypercube
sample forms the candidate set. Reachability (and UG-BEACON's guidance) uses
a non-uniform partition of the outcome plane that is finer in the
high-utility (high ``q``, high ``sigma``) region.
"""

from __future__ import annotations

import numpy as np
import torch

from beacon.metrics import BoxPartition, Reachability
from beacon.problems.base import DiscreteProblem

# Physical bounds of the 7 process variables (float32, as in the original script).
OIL_LOWER = torch.tensor([3 / 7, 0.7, 12, 0.12, 0, 16, 0.41])
OIL_UPPER = torch.tensor([7 / 3, 2, 24, 0.18, 0.2, 28, 1.32])


def latin_hypercube(n_factors: int, n_samples: int, random_state: int = 0) -> np.ndarray:
    """Classic randomized Latin hypercube on ``[0, 1]^d``.

    Bit-for-bit equivalent to ``pyDOE2.lhs(n_factors, n_samples,
    random_state=random_state)`` (no criterion); pyDOE2 itself no longer imports
    on Python >= 3.12.
    """
    rng = np.random.RandomState(random_state)
    cut = np.linspace(0, 1, n_samples + 1)
    u = rng.rand(n_samples, n_factors)
    a, b = cut[:n_samples], cut[1 : n_samples + 1]
    rdpoints = np.zeros_like(u)
    for j in range(n_factors):  # column-wise, exactly as pyDOE2 (bit-identical results)
        rdpoints[:, j] = u[:, j] * (b - a) + a
    H = np.zeros_like(rdpoints)
    for j in range(n_factors):
        order = rng.permutation(range(n_samples))
        H[:, j] = rdpoints[order, j]
    return H


def oil_sorbent(X: torch.Tensor) -> torch.Tensor:
    """Response surfaces for (q, sigma) given physical process variables ``(n, 7)``."""
    V1, V2, V3, V4, V5, V6, V7 = torch.split(X, 1, -1)
    q = (
        -212.8531 + 245.7998 * V1 - 127.3395 * V2 + 305.8461 * V3 + 638.1605 * V4 + 301.2118 * V5
        - 451.3796 * V6 - 115.5485 * V7 + 42.8351 * V1 * V2 + 262.3775 * V1 * V3 - 103.5274 * V1 * V4
        - 196.1568 * V1 * V5 - 394.7975 * V1 * V6 - 176.3341 * V1 * V7 + 74.8291 * V2 * V3
        + 4.1557 * V2 * V4 - 133.8683 * V2 * V5 + 65.8711 * V2 * V6 - 42.6911 * V2 * V7
        - 323.9363 * V3 * V4 - 107.3983 * V3 * V5 - 323.2353 * V3 * V6 + 46.9172 * V3 * V7
        - 144.4199 * V4 * V5 + 272.3729 * V4 * V6 + 49.0799 * V4 * V7 + 318.4706 * V5 * V6
        - 236.2498 * V5 * V7 + 252.4848 * V6 * V7 - 286.0182 * V4.pow(2) + 393.5992 * V6.pow(2)
    )
    sigma = (
        7.7696 + 15.4344 * V1 - 10.6190 * V2 - 17.9367 * V3 + 17.1385 * V4 + 2.5026 * V5
        - 24.3010 * V6 + 10.6058 * V7 - 1.2041 * V1 * V2 - 37.2207 * V1 * V3 - 3.2265 * V1 * V4
        + 7.3121 * V1 * V5 + 52.3994 * V1 * V6 + 9.7485 * V1 * V7 - 15.9371 * V2 * V3
        - 1.1706 * V2 * V4 - 2.6297 * V2 * V5 + 7.0225 * V2 * V6 - 1.4938 * V2 * V7
        + 30.2786 * V3 * V4 + 14.5061 * V3 * V5 + 48.5021 * V3 * V6 - 11.4857 * V3 * V7
        - 3.1381 * V4 * V5 - 14.9747 * V4 * V6 + 4.5204 * V4 * V7 - 17.6907 * V5 * V6
        - 19.2489 * V5 * V7 - 9.8219 * V6 * V7 - 18.7356 * V1.pow(2) + 12.1928 * V2.pow(2)
        - 17.5460 * V4.pow(2) + 5.4997 * V5.pow(2) - 26.2718 * V6.pow(2)
    )
    return torch.cat([q, sigma], dim=-1)


def _grid_boxes(lower, upper, partitions) -> BoxPartition:
    # float32 edges reproduce the original `partition_bounds` exactly.
    edges = [torch.linspace(float(lo), float(hi), p + 1) for lo, hi, p in zip(lower, upper, partitions)]
    lb, ub = [], []
    for i in range(partitions[0]):
        for j in range(partitions[1]):
            lb.append(torch.stack([edges[0][i], edges[1][j]]))
            ub.append(torch.stack([edges[0][i + 1], edges[1][j + 1]]))
    return BoxPartition(torch.stack(lb).double(), torch.stack(ub).double())


def oil_partition(mins: torch.Tensor, maxs: torch.Tensor, cut_q: float = 60000, cut_sigma: float = 6000) -> BoxPartition:
    """Non-uniform partition: 4x4 (upper left), 1 (lower left), 4x4 (lower right), 5x5 (upper right)."""
    q_lo, s_lo = float(mins[0]), float(mins[1])
    q_hi, s_hi = float(maxs[0]), float(maxs[1])
    parts = [
        _grid_boxes([q_lo, cut_sigma], [cut_q, s_hi], [4, 4]),
        _grid_boxes([q_lo, s_lo], [cut_q, cut_sigma], [1, 1]),
        _grid_boxes([cut_q, s_lo], [q_hi, cut_sigma], [4, 4]),
        _grid_boxes([cut_q, cut_sigma], [q_hi, s_hi], [5, 5]),
    ]
    return BoxPartition.concat(parts, clip_lower=mins.double(), clip_upper=maxs.double())


def make_oil_sorbent(n_candidates: int = 5000, lhs_seed: int = 0, **_) -> DiscreteProblem:
    """Oil-sorbent candidate set (7 inputs, 2 outcomes) with the paper's partition."""
    X = torch.as_tensor(latin_hypercube(7, n_candidates, random_state=lhs_seed), dtype=torch.double)
    Y = oil_sorbent(OIL_LOWER + (OIL_UPPER - OIL_LOWER) * X)
    Y[Y < 0] = 0
    partition = oil_partition(Y.min(dim=0).values, Y.max(dim=0).values)
    defaults = dict(
        n_init=10,
        iterations=200,
        replicates=10,
        k=3,
        kernel="rbf",
        likelihood="gamma",
        archive="observed",
        discrete_sampler="rff",
    )
    return DiscreteProblem(
        "oil-sorbent",
        X,
        Y,
        metric=Reachability.from_dataset(partition, Y),
        partition=partition,
        defaults=defaults,
        description="Oil sorbent (7 process variables -> capacity, strength), non-uniform 58-region partition.",
    )
