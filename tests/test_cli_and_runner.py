"""End-to-end runs through the Python API and the command line, on each device."""

import json
import subprocess
import sys

import pytest
import torch

from beacon import ContinuousProblem, register_problem, run_experiment
from beacon.cli import main
from beacon.device import resolve_device
from beacon.metrics import HistogramPartition, Reachability
from conftest import REPO_ROOT

FAST = {"num_restarts": 2, "raw_samples": 16, "maxiter": 20, "num_bases": 128}


@pytest.mark.parametrize("algorithm", ["beacon", "ug-beacon"])
def test_run_experiment_continuous(device, algorithm, tmp_path):
    overrides = {"iterations": 3, "replicates": 2, "n_init": 5, "pool_size": 128, **FAST}
    result = run_experiment(
        "ackley", algorithm, problem_kwargs={"dim": 2}, overrides=overrides,
        device=device, output_dir=tmp_path, verbose=0,
    )
    assert result.metric.shape == (2, 4) and result.cost.shape == (2, 4)
    assert torch.all(result.metric[:, 1:] >= result.metric[:, :-1])  # reachability never decreases
    assert torch.equal(result.cost[0], torch.arange(4, dtype=torch.float32))
    for name in ("cost.pt", "coverage.pt", "time.pt", "history.pt", "config.json"):
        assert (tmp_path / name).exists()
    config = json.loads((tmp_path / "config.json").read_text())
    assert config["algorithm"] == algorithm and config["device"].startswith(device.type)
    history = torch.load(tmp_path / "history.pt", weights_only=False)
    assert history[0]["X"].shape == (8, 2)


@pytest.mark.parametrize("algorithm", ["beacon", "ug-beacon"])
def test_run_experiment_discrete(device, algorithm):
    result = run_experiment(
        "water-solubility", algorithm, overrides={"iterations": 3, "replicates": 1},
        device=device, output_dir=None, verbose=0,
    )
    idx = result.histories[0]["indices"]
    assert len(idx) == 13 and len(set(idx.tolist())) == 13  # never re-selects a candidate


def test_seeded_runs_are_reproducible():
    kwargs = dict(problem_kwargs={"dim": 2}, overrides={"iterations": 3, "replicates": 1, **FAST},
                  device="cpu", output_dir=None, verbose=0)
    a = run_experiment("ackley", "beacon", **kwargs)
    b = run_experiment("ackley", "beacon", **kwargs)
    assert torch.allclose(a.histories[0]["X"], b.histories[0]["X"])


def test_early_stop_pads_full_reachability():
    part = HistogramPartition(0, 1, 1)
    problem = ContinuousProblem(
        "unit", lambda X: X[:, :1], 1, 1, 0.0, 1.0, metric=Reachability(part), partition=part,
        defaults={"n_init": 3, "iterations": 4, "replicates": 1},
    )
    result = run_experiment(problem, "beacon", overrides=FAST, device="cpu", output_dir=None, verbose=0)
    assert result.metric.tolist() == [[1.0] * 5]
    assert result.histories[0]["X"].shape[0] == 3  # no evaluation spent after full coverage


def test_custom_problem_registration():
    def factory(**_):
        part = HistogramPartition(-1, 1, 10)
        return ContinuousProblem(
            "my-sine", lambda X: torch.sin(3 * X).sum(-1, keepdim=True), 2, 1, -1.0, 1.0,
            metric=Reachability(part), partition=part, defaults={"n_init": 4, "iterations": 2, "replicates": 1},
        )

    register_problem("my-sine", factory, "test problem")
    result = run_experiment("my-sine", "beacon", overrides=FAST, device="cpu", output_dir=None, verbose=0)
    assert result.metric.shape == (1, 3)


def test_unknown_setting_is_rejected():
    with pytest.raises(ValueError, match="Unknown setting"):
        run_experiment("ackley", "beacon", overrides={"iteration": 3}, output_dir=None, verbose=0)


def test_resolve_device():
    assert resolve_device("cpu").type == "cpu"
    assert resolve_device("auto").type == ("cuda" if torch.cuda.is_available() else "cpu")
    if not torch.cuda.is_available():
        with pytest.raises(ValueError, match="CUDA is not available"):
            resolve_device("cuda")
    with pytest.raises(ValueError, match="mps"):
        resolve_device("mps")


def test_cli_run_and_list(tmp_path, capsys):
    assert main(["list"]) == 0
    out = capsys.readouterr().out
    assert "beacon-saas" in out and "oil-sorbent" in out and "synthetic" in out
    code = main([
        "run", "--problem", "rosenbrock", "--dim", "2", "-a", "UG_BEACON", "--iterations", "2",
        "--replicates", "1", "--device", "cpu", "--output-dir", str(tmp_path), "-v", "0",
        "--set", "num_restarts=2", "raw_samples=16", "pool_size=64", "k=3",
    ])
    assert code == 0
    config = json.loads((tmp_path / "config.json").read_text())
    assert config["algorithm"] == "ug-beacon" and config["beacon"]["k"] == 3


def test_cli_config_file(tmp_path):
    cfg = tmp_path / "run.json"
    cfg.write_text(json.dumps({
        "problem": "ackley", "problem_kwargs": {"dim": 2}, "algorithm": "beacon",
        "overrides": {"iterations": 1, "replicates": 1, **FAST}, "output_dir": str(tmp_path / "out"),
    }))
    assert main(["run", "--config", str(cfg), "--device", "cpu", "-v", "0"]) == 0
    assert (tmp_path / "out" / "coverage.pt").exists()


def test_cli_reports_bad_input(capsys):
    assert main(["run", "--problem", "ackley", "-a", "random", "--no-save"]) == 2
    assert "Unknown algorithm" in capsys.readouterr().err


def test_experiment_wrapper_script_runs(tmp_path):
    script = REPO_ROOT / "experiments" / "continuous" / "single-outcome" / "ackley-beacon.py"
    result = subprocess.run(
        [sys.executable, str(script), "--dim", "2", "--iterations", "1", "--replicates", "1",
         "--device", "cpu", "--no-save", "--set", "num_restarts=2", "raw_samples=16"],
        cwd=tmp_path, capture_output=True, text=True, check=True,
    )
    assert "final reachability" in result.stdout
