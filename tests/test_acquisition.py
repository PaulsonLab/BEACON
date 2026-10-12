"""Checks for the k-NN novelty acquisition and discrete Thompson sampling."""

import pytest
import torch

from beacon.acquisition import (
    BeaconAcquisition,
    candidate_scores,
    draw_discrete_samples,
    knn_novelty,
    resolve_discrete_method,
)
from beacon.models import build_independent_gps
from conftest import toy_data


def original_knn(samples, archive, k):
    """The original scripts' sort-and-mask formulation."""
    dist = torch.cdist(samples, archive)
    dist, _ = torch.sort(dist, dim=1)
    n = dist.shape[1]
    E = torch.cat((torch.ones(k, dtype=dist.dtype), torch.zeros(n - k, dtype=dist.dtype)))
    return (dist * E).sum(dim=1)


@pytest.mark.parametrize("m", [1, 2, 5])
def test_knn_novelty_matches_original_formulation(m):
    g = torch.Generator().manual_seed(0)
    samples, archive = torch.randn(30, m, generator=g), torch.randn(17, m, generator=g)
    assert torch.allclose(knn_novelty(samples, archive, 4), original_knn(samples, archive, 4))


def test_knn_novelty_clips_k_and_validates():
    samples, archive = torch.randn(3, 2), torch.randn(2, 2)
    assert torch.allclose(knn_novelty(samples, archive, 10), torch.cdist(samples, archive).sum(-1))
    with pytest.raises(ValueError):
        knn_novelty(samples, torch.randn(2, 3), 1)
    with pytest.raises(ValueError):
        knn_novelty(samples, torch.empty(0, 2), 1)


def test_knn_novelty_is_zero_on_archive_and_grows_away():
    archive = torch.tensor([[0.0], [1.0], [2.0]])
    scores = knn_novelty(torch.tensor([[1.0], [10.0]]), archive, 1)
    assert scores[0] == 0 and scores[1] == 8


def test_beacon_acquisition_shapes_and_gradients(fitted_model, device):
    model, X, Y = fitted_model
    acq = BeaconAcquisition(model, Y, k=3, num_bases=256)
    Xq = torch.rand(7, 1, 2, dtype=torch.double, device=device, requires_grad=True)
    values = acq(Xq)
    assert values.shape == (7,) and values.device.type == device.type
    values.sum().backward()
    assert torch.isfinite(Xq.grad).all()
    # The sample path is fixed: repeated evaluation is deterministic.
    assert torch.equal(acq(Xq).detach(), values.detach())


def test_multi_output_acquisition(device):
    torch.manual_seed(0)
    X, Y = toy_data(m=2, device=device)
    model = build_independent_gps(X, Y)
    acq = BeaconAcquisition(model, Y, k=2, num_bases=128)
    out = acq.sample_outcomes(torch.rand(4, 1, 2, dtype=torch.double, device=device))
    assert out.shape == (4, 1, 2)


@pytest.mark.parametrize("method", ["exact", "rff"])
def test_discrete_samples_shape(fitted_model, device, method):
    model, _, _ = fitted_model
    cand = torch.rand(50, 2, dtype=torch.double, device=device)
    samples = draw_discrete_samples(model, cand, method=method, num_bases=128, chunk_size=16)
    assert samples.shape == (50, 1) and samples.device.type == device.type
    assert torch.isfinite(samples).all()


def test_exact_and_rff_samples_agree_in_distribution(fitted_model):
    model, _, _ = fitted_model
    cand = torch.rand(5, 2, dtype=torch.double)
    exact = torch.stack([draw_discrete_samples(model, cand, method="exact") for _ in range(400)])
    rff = torch.stack([draw_discrete_samples(model, cand, method="rff") for _ in range(400)])
    sd = exact.std(0)
    assert torch.all((exact.mean(0) - rff.mean(0)).abs() < 0.3 * sd + 1e-6)
    assert torch.all((rff.std(0) / sd - 1).abs() < 0.25)


def test_resolve_discrete_method():
    assert resolve_discrete_method("exact", 10**6, 10) == "exact"
    assert resolve_discrete_method("auto", 100, 1000) == "exact"
    assert resolve_discrete_method("auto", 5000, 1000) == "rff"


def test_candidate_scores_chunking():
    s, a = torch.randn(100, 2), torch.randn(9, 2)
    assert torch.allclose(candidate_scores(s, a, 3, chunk_size=7), knn_novelty(s, a, 3))
