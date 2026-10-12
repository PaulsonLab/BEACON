"""Reachability metrics and outcome partitions."""

import numpy as np
import pytest
import torch

from beacon.metrics import (
    BestValue,
    BoxPartition,
    ClassifierPartition,
    GridPartition,
    HistogramPartition,
    Reachability,
    uniformity,
)


def original_reachability(behavior, n_bins, lo, hi):
    """Single-outcome reachability exactly as in the original scripts."""
    hist, _ = np.histogram(behavior.squeeze(1).numpy(), np.linspace(lo, hi, n_bins + 1))
    return len(hist[np.nonzero(hist)]) / n_bins


@pytest.mark.parametrize("seed", range(5))
def test_histogram_reachability_matches_numpy(seed):
    g = torch.Generator().manual_seed(seed)
    y = torch.randn(200, 1, generator=g, dtype=torch.double) * 3
    y[0] = 4.0  # exactly on the upper edge -> last bin
    y[1] = -9.0  # outside -> ignored
    metric = Reachability(HistogramPartition(-4.0, 4.0, 25))
    assert metric(y) == pytest.approx(original_reachability(y, 25, -4.0, 4.0))


def test_histogram_partition_edges():
    part = HistogramPartition(0, 1, 4)
    idx = part.region_index(torch.tensor([[0.0], [0.25], [0.999], [1.0], [1.01], [-0.1]], dtype=torch.double))
    assert idx.tolist() == [0, 1, 3, 3, -1, -1]


def test_grid_partition_clamps_upper_edge():
    part = GridPartition([0, 0], [1, 1], 10)
    idx = part.region_index(torch.tensor([[1.0, 0.55], [0.0, 0.0], [1.2, 0.5]], dtype=torch.double))
    assert idx.tolist() == [9 * 10 + 5, 0, -1]
    assert part.n_regions == 100


def test_reachability_from_dataset_normalizes_by_reachable_regions():
    Y = torch.tensor([[0.0], [0.1], [0.9], [1.0]], dtype=torch.double)
    metric = Reachability.from_dataset(HistogramPartition(0, 1, 10), Y)
    assert metric.normalizer == 3  # bins 0, 1 and 9 are occupied
    assert metric(Y[:1]) == pytest.approx(1 / 3) and metric(Y) == 1.0


def test_box_partition_from_grid_and_overlap():
    part = BoxPartition.from_grid([0, 0], [2, 2], [2, 2])
    assert part.n_regions == 4
    member = part.membership(torch.tensor([[0.5, 0.5], [1.0, 1.0]], dtype=torch.double))
    assert member[0].sum() == 1
    assert member[1].sum() == 4  # closed boxes share the corner point


def test_box_partition_clip():
    part = BoxPartition(torch.tensor([[0.0, 0.0]]), torch.tensor([[1.0, 1.0]]), clip_lower=[0, 0], clip_upper=[1, 1])
    assert part.membership(torch.tensor([[5.0, -3.0]], dtype=torch.double)).all()


def test_classifier_partition_threshold():
    def classify(Y):
        return Y[:, 0].long(), Y[:, 1]

    part = ClassifierPartition(classify, 3, threshold=0.9)
    Y = torch.tensor([[0, 0.95], [2, 0.5], [1, 0.99]], dtype=torch.double)
    # class 2 is predicted with confidence 0.5 < 0.9, so it is not reached
    assert part.visited(Y).tolist() == [True, True, False]


def test_uniformity_and_best_value():
    part = HistogramPartition(0, 1, 2)
    even = torch.tensor([[0.1], [0.9]], dtype=torch.double)
    skewed = torch.tensor([[0.1], [0.2], [0.3], [0.9]], dtype=torch.double)
    assert uniformity(even, part) == pytest.approx(1.0)
    assert uniformity(skewed, part) < 1.0
    assert BestValue(lambda Y: -Y[:, 0])(even) == pytest.approx(-0.1)
