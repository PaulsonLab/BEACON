"""Algorithm selection interface and one-step behaviour of each variant."""

import pytest
import torch

from beacon import ALGORITHMS, BEACON, BEACONSAAS, UGBEACON, BeaconConfig, get_algorithm
from beacon.algorithms import canonical_name
from beacon.metrics import GridPartition, HistogramPartition
from conftest import toy_data

FAST = dict(num_restarts=2, raw_samples=16, maxiter=20, num_bases=128)
FAST_SAAS = dict(warmup_steps=16, num_samples=16, thinning=8)


def test_registry_has_three_algorithms():
    assert set(ALGORITHMS) == {"beacon", "ug-beacon", "beacon-saas"}


@pytest.mark.parametrize(
    "alias,expected",
    [("BEACON", "beacon"), ("ug_beacon", "ug-beacon"), ("UG-BEACON", "ug-beacon"),
     ("beacon_saas", "beacon-saas"), ("saas", "beacon-saas")],
)
def test_aliases(alias, expected):
    assert canonical_name(alias) == expected


def test_get_algorithm_types_and_errors():
    assert type(get_algorithm("beacon")) is BEACON
    assert isinstance(get_algorithm("beacon-saas"), BEACONSAAS)
    assert get_algorithm("beacon-saas").config.surrogate == "saas"
    assert isinstance(get_algorithm("ug-beacon", partition=HistogramPartition(0, 1, 5)), UGBEACON)
    with pytest.raises(ValueError, match="Unknown algorithm"):
        get_algorithm("random-search")
    with pytest.raises(ValueError, match="partition"):
        get_algorithm("ug-beacon")


def test_config_validation_and_overrides():
    algo = get_algorithm("beacon", BeaconConfig(k=5), kernel="matern")
    assert algo.config.k == 5 and algo.config.kernel == "matern"
    with pytest.raises(ValueError):
        BeaconConfig(kernel="linear")
    with pytest.raises(ValueError, match="Unknown BEACON option"):
        BeaconConfig().replace(not_an_option=1)


def _bounds(d, device):
    return torch.stack([torch.zeros(d), torch.ones(d)]).to(dtype=torch.double, device=device)


@pytest.mark.parametrize("archive", ["observed", "posterior_mean"])
def test_beacon_continuous_step(device, archive):
    torch.manual_seed(0)
    X, Y = toy_data(device=device)
    algo = get_algorithm("beacon", archive=archive, k=3, **FAST)
    x = algo.suggest(X, Y, bounds=_bounds(2, device))
    assert x.shape == (1, 2) and x.device.type == device.type
    assert torch.all((x >= 0) & (x <= 1))


def test_beacon_discrete_step_never_repeats(device):
    torch.manual_seed(0)
    X, Y = toy_data(n=30, device=device)
    available = torch.ones(30, dtype=torch.bool, device=device)
    available[:10] = False
    algo = get_algorithm("beacon", k=3)
    for _ in range(3):
        idx = algo.suggest(X[~available], Y[~available], candidates=X, available=available)
        assert available[idx]
        available[idx] = False


def test_beacon_prefers_novel_outcomes():
    """With a dense archive at low outcomes, BEACON should pick a high-outcome candidate."""
    torch.manual_seed(0)
    X = torch.linspace(0, 1, 40, dtype=torch.double).unsqueeze(-1)
    Y = 10 * X
    observed = torch.arange(0, 15)  # outcomes in [0, 3.6]
    available = torch.ones(40, dtype=torch.bool)
    available[observed] = False
    algo = get_algorithm("beacon", k=3, kernel="rbf")
    idx = algo.suggest(X[observed], Y[observed], candidates=X, available=available)
    assert idx >= 30


def test_ug_beacon_avoids_visited_regions_and_falls_back():
    torch.manual_seed(0)
    X = torch.linspace(0, 1, 40, dtype=torch.double).unsqueeze(-1)
    Y = 10 * X
    observed = torch.tensor([0, 1, 2, 39])
    available = torch.ones(40, dtype=torch.bool)
    available[observed] = False
    # Regions [0,5) and [5,10]; only the upper region's top end was observed.
    algo = get_algorithm("ug-beacon", partition=HistogramPartition(0, 10, 2), k=1)
    idx = algo.suggest(X[observed], Y[observed], candidates=X, available=available)
    assert algo.last_fallback  # both regions visited -> unconstrained BEACON choice
    assert available[idx]

    # Exclude nothing but give 4 regions: the selected sample must land in an unvisited one.
    algo = get_algorithm("ug-beacon", partition=HistogramPartition(0, 10, 4), k=1)
    algo.suggest(X[observed], Y[observed], candidates=X, available=available)
    assert not algo.last_fallback


def test_ug_beacon_continuous_pool(device):
    torch.manual_seed(0)
    X, Y = toy_data(m=2, device=device)
    part = GridPartition(Y.min(0).values.cpu() - 1, Y.max(0).values.cpu() + 1, 5)
    algo = get_algorithm("ug-beacon", partition=part, pool_size=256, **FAST)
    x = algo.suggest(X, Y, bounds=_bounds(2, device))
    assert x.shape == (1, 2) and torch.all((x >= 0) & (x <= 1))


@pytest.mark.slow
@pytest.mark.parametrize("mode", ["random", "mean"])
def test_beacon_saas_discrete_step(mode):
    require_saas_backend()
    torch.manual_seed(0)
    X, Y = toy_data(n=20, d=4)
    available = torch.ones(20, dtype=torch.bool)
    available[:8] = False
    algo = get_algorithm("beacon-saas", saas_sample_mode=mode, **FAST_SAAS)
    idx = algo.suggest(X[:8], Y[:8], candidates=X, available=available)
    assert idx >= 8


@pytest.mark.slow
def test_beacon_saas_continuous_step():
    require_saas_backend()
    torch.manual_seed(0)
    X, Y = toy_data(n=10, d=3)
    algo = get_algorithm("beacon-saas", **FAST_SAAS, **FAST)
    x = algo.suggest(X, Y, bounds=_bounds(3, "cpu"))
    assert x.shape == (1, 3)


def require_saas_backend():
    """Skip when BoTorch's fully Bayesian backend (JAX/NumPyro or Pyro) is missing."""
    from botorch.models.fully_bayesian import SaasFullyBayesianSingleTaskGP

    try:
        SaasFullyBayesianSingleTaskGP(torch.rand(3, 1, dtype=torch.double), torch.rand(3, 1, dtype=torch.double))
    except ImportError as exc:
        pytest.skip(f"SAAS backend unavailable: {exc}")
