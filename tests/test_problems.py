"""Benchmark definitions reproduce the paper's experimental settings.

The initial designs and reachability metrics are checked against the tracked
results that produced the paper figures (``results/``): for every replicate
seed, the reachability of the initial design must match the saved value.
"""

import numpy as np
import pytest
import torch
from scipy.io import loadmat

from beacon.paths import CONTINUOUS_SINGLE_OUTCOME_RESULTS_DIR, PLOT_DATA_DIR
from beacon.problems import get_problem, list_problems
from beacon.problems.oil_sorbent import latin_hypercube

SYNTHETIC = {"ackley": "Ackley", "rosenbrock": "Rosen", "styblinski-tang": "StyTang"}


def initial_metric(problem, n_init, seeds):
    if problem.is_discrete:
        return np.array([problem.metric(problem.Y_true[problem.initial_indices(n_init, s)]) for s in seeds])
    out = []
    for s in seeds:
        _, _, y_true = problem.initial_data(n_init, s)
        out.append(problem.metric(y_true))
    return np.array(out)


@pytest.mark.parametrize("dim", [4, 8, 12])
@pytest.mark.parametrize("name", sorted(SYNTHETIC))
def test_synthetic_initial_reachability_matches_paper_results(name, dim):
    problem = get_problem(name, dim=dim)
    tag = f"{dim}D{SYNTHETIC[name]}"
    saved = torch.load(CONTINUOUS_SINGLE_OUTCOME_RESULTS_DIR / tag / f"{tag}_coverage_list_BEACON.pt")
    assert problem.defaults["n_init"] == 10 and problem.defaults["iterations"] == saved.shape[1] - 1
    assert problem.defaults["replicates"] == saved.shape[0]
    np.testing.assert_allclose(initial_metric(problem, 10, range(20)), saved[:, 0].numpy(), atol=1e-6)


def test_noisy_ackley_settings_match_paper_results():
    problem = get_problem("noisy-ackley", noise_std=1.0)
    saved = torch.load(
        CONTINUOUS_SINGLE_OUTCOME_RESULTS_DIR / "Noise" / "4DAckley" / "4DNoisyAckley_coverage_list_BEACON_considernoise_1.0.pt"
    )
    assert problem.defaults["iterations"] == saved.shape[1] - 1
    np.testing.assert_allclose(initial_metric(problem, 50, range(20)), saved[:, 0].numpy(), atol=1e-6)


@pytest.mark.parametrize(
    "name,mat,key",
    [
        ("hydrogen", "MOF.mat", "coverage_NS_TS1_0"),
        ("nitrogen", "MOF.mat", "coverage_NS_TS1_1"),
        ("water-solubility", "SmallMolecule.mat", "coverage_NS_TS1_0"),
        ("logd", "SmallMolecule.mat", "coverage_NS_TS1_2"),
        ("logd", "SmallMolecule.mat", "coverage_BEACON_SAAS_2"),
    ],
)
def test_material_initial_reachability_matches_paper_results(name, mat, key):
    problem = get_problem(name)
    saved = loadmat(PLOT_DATA_DIR / mat)[key]
    assert problem.defaults["iterations"] == saved.shape[1] - 1
    np.testing.assert_allclose(initial_metric(problem, 10, range(20)), saved[:, 0], atol=1e-5)


def test_joint_gas_grid_fix_only_changes_normalizer():
    """The original grid counted 2 zero-width edge cells (64 instead of 63 regions)."""
    problem = get_problem("joint-gas")
    assert problem.metric.normalizer == 63
    saved = loadmat(PLOT_DATA_DIR / "MOF.mat")["coverage_NS_TS1_2"][:, 0]
    np.testing.assert_allclose(initial_metric(problem, 50, range(20)) * 63 / 64, saved, atol=1e-5)
    assert problem.input_dim == 25


def test_latin_hypercube_matches_pydoe2():
    expected = np.array([
        [0.6252801159149844, 0.49449737799915067, 0.3846239676026084],
        [0.4308662944577353, 0.19497969806002866, 0.5671097450254678],
        [0.87497062520639, 0.5180127833399404, 0.10960230786022337],
        [0.019077072343489293, 0.9198074990302352, 0.9509347590260939],
    ])
    np.testing.assert_array_equal(latin_hypercube(3, 4, random_state=7), expected)
    H = latin_hypercube(7, 5000, random_state=0)
    np.testing.assert_array_equal(
        H[0], [0.7387483492343654, 0.6760678350234453, 0.1478197911302516, 0.9450043023198968,
               0.49585178537338953, 0.6966953424891276, 0.9796963045884545],
    )


def test_oil_sorbent_partition():
    problem = get_problem("oil-sorbent")
    assert problem.n_candidates == 5000 and problem.output_dim == 2
    assert problem.partition.n_regions == 58
    assert problem.metric.normalizer == 42
    assert problem.defaults["k"] == 3 and problem.defaults["replicates"] == 10
    assert (problem.Y >= 0).all()


def test_mnist_problem():
    problem = get_problem("mnist")
    assert problem.X.shape == (2000, 8) and problem.Y.shape == (2000, 196)
    assert problem.output_shape == (14, 14)
    assert problem.defaults["surrogate"] == "hogp"
    # Seeds 0 and 1 match the saved paper data (4 and 3 confidently classified digits).
    assert [round(10 * v) for v in initial_metric(problem, 10, [0, 1])] == [4, 3]
    assert problem.guidance_partition is not problem.partition


def test_multi_output_plus_problem():
    problem = get_problem("multi-output-plus")
    X = problem.initial_design(10, 0)
    y_obs, y_true = problem.evaluate(X)
    assert y_obs.shape == (10, 2) and torch.equal(y_obs, y_true)
    assert problem.partition.n_regions == 100 and problem.defaults["iterations"] == 500


def test_noisy_problem_adds_noise_but_metric_uses_truth():
    problem = get_problem("noisy-ackley", noise_std=1.0)
    torch.manual_seed(0)
    X = problem.initial_design(20, 0)
    y_obs, y_true = problem.evaluate(X)
    assert not torch.allclose(y_obs, y_true)
    assert (y_obs - y_true).std() == pytest.approx(1.0, abs=0.4)


def test_registry_lists_all_paper_problems():
    names = set(list_problems())
    assert {"ackley", "rosenbrock", "styblinski-tang", "noisy-ackley", "multi-output-plus", "maze",
            "hydrogen", "nitrogen", "joint-gas", "water-solubility", "esol", "logd", "oil-sorbent",
            "mnist"} <= names
    with pytest.raises(ValueError, match="Unknown problem"):
        get_problem("nope")
