"""Focused checks for quiet continuous optimization helpers."""

import importlib.util
import os
import warnings
from pathlib import Path

import numpy as np
import pytest
import torch
from botorch.models import SingleTaskGP
from botorch.models.transforms.outcome import Standardize
from botorch.test_functions import Ackley
from gpytorch.kernels import RBFKernel, ScaleKernel
from gpytorch.mlls import ExactMarginalLogLikelihood

from beacon.optimization import (
    DTYPE,
    fit_gpytorch_mll_quiet,
    optimize_acqf_quiet,
    report_warnings,
    run_with_warning_capture,
    validate_acquisition_result,
    warning_stage_summary,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


def load_ackley_module():
    os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/mplcache")
    path = REPO_ROOT / "experiments" / "continuous" / "single-outcome" / "ackley-beacon.py"
    spec = importlib.util.spec_from_file_location("ackley_beacon", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_warning_capture_distinguishes_fit_and_acquisition_stages():
    def warn(message):
        def callback():
            warnings.warn(message, RuntimeWarning)
            return message

        return callback

    _, fit_warnings = run_with_warning_capture(warn("fit warning"))
    _, opt_warnings = run_with_warning_capture(warn("optimizer warning"))

    fit_summary = warning_stage_summary("gp_fit", fit_warnings)
    opt_summary = warning_stage_summary("acquisition_optimization", opt_warnings)

    assert fit_summary["stage"] == "gp_fit"
    assert opt_summary["stage"] == "acquisition_optimization"
    assert fit_summary["first_message"] == "fit warning"
    assert opt_summary["first_message"] == "optimizer warning"


def test_warning_reporting_is_quiet_by_default(capsys, monkeypatch):
    monkeypatch.delenv("BEACON_DEBUG_OPT", raising=False)

    _, warning_records = run_with_warning_capture(
        lambda: warnings.warn("expected optimizer retry", RuntimeWarning)
    )
    report_warnings("acquisition_optimization", warning_records, seed=0, iteration=0)

    assert capsys.readouterr().out == ""


def test_warning_reporting_prints_details_in_debug_mode(capsys, monkeypatch):
    monkeypatch.setenv("BEACON_DEBUG_OPT", "1")

    _, warning_records = run_with_warning_capture(
        lambda: warnings.warn("expected optimizer retry", RuntimeWarning)
    )
    report_warnings("acquisition_optimization", warning_records, seed=0, iteration=0)

    output = capsys.readouterr().out
    assert "BEACON debug" in output
    assert "expected optimizer retry" in output


def test_invalid_candidate_validation_raises_clear_error():
    bounds = torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=DTYPE)
    candidate = torch.tensor([[1.5, 0.5]], dtype=DTYPE)
    acq_value = torch.tensor([1.0], dtype=DTYPE)

    with pytest.raises(RuntimeError, match="Candidate outside"):
        validate_acquisition_result(candidate, acq_value, bounds, seed=0, iteration=0)


def test_ackley_acquisition_step_returns_finite_in_bounds():
    module = load_ackley_module()
    torch.manual_seed(7)
    np.random.seed(7)

    dim = 2
    n_init = 6
    function = Ackley(dim=dim)
    train_x = torch.tensor(np.random.rand(n_init, dim), dtype=DTYPE)
    train_y = function(-5 + 10 * train_x).unsqueeze(1).to(dtype=DTYPE)

    covar_module = ScaleKernel(RBFKernel(ard_num_dims=dim)).to(dtype=DTYPE)
    model = SingleTaskGP(
        train_x,
        train_y,
        outcome_transform=Standardize(m=1),
        covar_module=covar_module,
    )
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_mll_quiet(mll, model=model, seed=0, iteration=0)

    model.train_x = train_x
    model.train_y = train_y
    acquisition = module.CustomAcquisitionFunction(model, train_y, k=3)
    bounds = torch.tensor([[0.0] * dim, [1.0] * dim], dtype=DTYPE)

    candidate, acq_value = optimize_acqf_quiet(
        acq_function=acquisition,
        bounds=bounds,
        q=1,
        num_restarts=2,
        raw_samples=8,
        options={"maxiter": 50, "batch_limit": 2},
        seed=0,
        iteration=0,
        model=model,
    )

    assert candidate.dtype == DTYPE
    assert torch.isfinite(acq_value).all()
