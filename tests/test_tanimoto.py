"""The Tanimoto kernel as a BEACON option (fingerprints / non-negative features)."""

import warnings

import pytest
import torch

from beacon import BeaconConfig, get_algorithm, run_experiment
from beacon.acquisition import resolve_discrete_method
from beacon.metrics import GridPartition, HistogramPartition
from beacon.models import TanimotoKernel, build_independent_gps
from beacon.thompson_sampling import pathwise_supported


def fingerprints(n=40, bits=32, seed=0, device="cpu"):
    """Random binary fingerprints with a smooth-ish scalar property."""
    g = torch.Generator().manual_seed(seed)
    X = (torch.rand(n, bits, generator=g) < 0.3).double()
    w = torch.linspace(-1, 1, bits, dtype=torch.double)
    Y = (X @ w).unsqueeze(-1) + 0.1 * X.sum(-1, keepdim=True)
    return X.to(device), Y.to(device)


def test_kernel_matches_definition_and_is_psd():
    X, _ = fingerprints(n=25)
    K = TanimotoKernel()(X, X).to_dense()
    i, j = 3, 7
    inter = (X[i] * X[j]).sum()
    union = X[i].sum() + X[j].sum() - inter
    eps = TanimotoKernel.eps
    assert torch.isclose(K[i, j], (inter + eps) / (union + eps))
    assert torch.allclose(K.diagonal(), torch.ones(25, dtype=torch.double))
    assert torch.linalg.eigvalsh(K).min() > -1e-9


def test_kernel_zero_vectors_consistent_with_diag():
    X = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 1.0], [0.0, 0.0, 0.0]], dtype=torch.double)
    K = TanimotoKernel()(X, X).to_dense()
    assert K[0, 0] == 1 and K[0, 2] == 1 and K[0, 1] < 1e-6
    assert torch.equal(K.diagonal(), TanimotoKernel()(X, X, diag=True))


def test_negative_inputs_are_rejected():
    X, Y = fingerprints(n=10)
    with pytest.raises(ValueError, match="non-negative"):
        build_independent_gps(X - 0.5, Y, kernel="tanimoto")


def test_invalid_combinations_fail_early():
    with pytest.raises(ValueError, match="surrogate='gp'"):
        get_algorithm("beacon-saas", kernel="tanimoto")
    with pytest.raises(ValueError, match="random-Fourier"):
        BeaconConfig(kernel="tanimoto", discrete_sampler="rff")


def test_tanimoto_has_no_pathwise_sampler():
    X, Y = fingerprints(n=12)
    assert not pathwise_supported(build_independent_gps(X, Y, kernel="tanimoto"))
    assert pathwise_supported(build_independent_gps(X, Y, kernel="rbf"))


def test_auto_sampler_uses_exact_for_tanimoto_with_warning():
    X, Y = fingerprints(n=12)
    model = build_independent_gps(X, Y, kernel="tanimoto")
    assert resolve_discrete_method("auto", 100, 1000, model) == "exact"
    with pytest.warns(RuntimeWarning, match="exact joint sampling"):
        assert resolve_discrete_method("auto", 5000, 1000, model) == "exact"


@pytest.mark.parametrize("algorithm", ["beacon", "ug-beacon"])
def test_discrete_step_with_tanimoto(device, algorithm):
    torch.manual_seed(0)
    X, Y = fingerprints(device=device)
    available = torch.ones(len(X), dtype=torch.bool, device=device)
    available[:10] = False
    kwargs = {"partition": HistogramPartition(float(Y.min()), float(Y.max()), 5)} if algorithm == "ug-beacon" else {}
    algo = get_algorithm(algorithm, kernel="tanimoto", likelihood="plain", k=3, **kwargs)
    idx = algo.suggest(X[:10], Y[:10], candidates=X, available=available)
    assert 10 <= idx < len(X)


@pytest.mark.parametrize("algorithm", ["beacon", "ug-beacon"])
def test_continuous_step_with_tanimoto_uses_pool(device, algorithm):
    torch.manual_seed(0)
    g = torch.Generator().manual_seed(1)
    X = torch.rand(10, 3, generator=g, dtype=torch.double).to(device)
    Y = torch.stack([X.sum(-1), X[:, 0] - X[:, 1]], dim=-1)
    bounds = torch.stack([torch.zeros(3), torch.ones(3)]).to(dtype=torch.double, device=device)
    kwargs = {"partition": GridPartition([0, -1], [3, 1], 4)} if algorithm == "ug-beacon" else {}
    algo = get_algorithm(algorithm, kernel="tanimoto", pool_size=256, k=3, **kwargs)
    x = algo.suggest(X, Y, bounds=bounds)
    assert x.shape == (1, 3) and x.device.type == device.type
    assert torch.all((x >= 0) & (x <= 1))


def test_run_experiment_with_tanimoto_override():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = run_experiment(
            "water-solubility", "beacon", overrides={"kernel": "tanimoto", "iterations": 2, "replicates": 1},
            device="cpu", output_dir=None, verbose=0,
        )
    assert result.settings["beacon"]["kernel"] == "tanimoto"
    assert len(set(result.histories[0]["indices"].tolist())) == 12


def test_matches_gauche_kernel():
    gauche = pytest.importorskip("gauche.kernels.fingerprint_kernels.tanimoto_kernel")
    X, _ = fingerprints(n=15)
    ours = TanimotoKernel()(X, X).to_dense()
    theirs = gauche.TanimotoKernel()(X, X).to_dense()
    assert torch.allclose(ours, theirs, atol=1e-14)
