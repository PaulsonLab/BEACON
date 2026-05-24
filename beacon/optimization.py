"""Shared optimization helpers for continuous BEACON experiments."""

import os
import warnings

import torch
from botorch.fit import fit_gpytorch_mll
from botorch.optim import optimize_acqf


DTYPE = torch.double
ACQ_NUM_RESTARTS = 10
ACQ_RAW_SAMPLES = 64
ACQ_OPTIONS = {"maxiter": 200, "batch_limit": 5}


def debug_enabled():
    """Return whether optimizer warning details should be printed."""
    return os.environ.get("BEACON_DEBUG_OPT") == "1"


def run_with_warning_capture(callback):
    """Run a callable while capturing warnings instead of printing them."""
    with warnings.catch_warnings(record=True) as caught_warnings:
        warnings.simplefilter("always")
        result = callback()
    return result, caught_warnings


def warning_stage_summary(stage, warning_records):
    """Summarize captured warnings without dumping long SciPy retry messages."""
    if not warning_records:
        return {
            "stage": stage,
            "count": 0,
            "categories": [],
            "first_message": "",
        }

    first_message = str(warning_records[0].message).splitlines()[0]
    return {
        "stage": stage,
        "count": len(warning_records),
        "categories": sorted({record.category.__name__ for record in warning_records}),
        "first_message": first_message,
    }


def _iter_models(model):
    if model is None:
        return []
    return getattr(model, "models", [model])


def model_diagnostics(model):
    """Return compact GP hyperparameter diagnostics for debug messages."""
    noises = []
    lengthscales = []

    for submodel in _iter_models(model):
        likelihood = getattr(submodel, "likelihood", None)
        noise = getattr(likelihood, "noise", None)
        if torch.is_tensor(noise):
            noises.append(noise.detach().reshape(-1).cpu())

        covar_module = getattr(submodel, "covar_module", None)
        base_kernel = getattr(covar_module, "base_kernel", None)
        lengthscale = getattr(base_kernel, "lengthscale", None)
        if torch.is_tensor(lengthscale):
            lengthscales.append(lengthscale.detach().reshape(-1).cpu())

    return {
        "noise": torch.cat(noises) if noises else None,
        "lengthscale": torch.cat(lengthscales) if lengthscales else None,
    }


def _range_text(values):
    if values is None or values.numel() == 0:
        return "n/a"
    return f"[{float(values.min()):.3g}, {float(values.max()):.3g}]"


def _scalar_text(value):
    if value is None:
        return "n/a"
    if torch.is_tensor(value):
        return f"{float(value.detach().reshape(-1)[0].cpu()):.6g}"
    return f"{float(value):.6g}"


def report_warnings(stage, warning_records, seed=None, iteration=None, model=None, acq_value=None):
    """Print captured warnings only when BEACON_DEBUG_OPT=1."""
    if not warning_records or not debug_enabled():
        return

    summary = warning_stage_summary(stage, warning_records)
    diagnostics = model_diagnostics(model)
    print(
        "[BEACON debug] "
        f"seed={seed}, iter={iteration}, stage={stage}, "
        f"warnings={summary['count']} ({', '.join(summary['categories'])}), "
        f"candidate_value={_scalar_text(acq_value)}, "
        f"noise={_range_text(diagnostics['noise'])}, "
        f"lengthscale={_range_text(diagnostics['lengthscale'])}, "
        f"first={summary['first_message']}"
    )

    for warning_record in warning_records:
        print(
            "[BEACON debug detail] "
            f"{warning_record.category.__name__}: {warning_record.message}"
        )


def fit_gpytorch_mll_quiet(mll, *, model=None, seed=None, iteration=None):
    """Fit a GP model while capturing retry/noise warnings unless debug is enabled."""
    result, warning_records = run_with_warning_capture(lambda: fit_gpytorch_mll(mll))
    report_warnings("gp_fit", warning_records, seed=seed, iteration=iteration, model=model)
    return result


def validate_acquisition_result(candidate, acq_value, bounds, seed=None, iteration=None):
    """Fail clearly if acquisition optimization returns an unusable candidate."""
    acq_value_tensor = acq_value if torch.is_tensor(acq_value) else torch.as_tensor(acq_value)

    if not torch.isfinite(candidate).all():
        raise RuntimeError(f"Non-finite candidate at seed={seed}, iter={iteration}: {candidate}")
    if not torch.isfinite(acq_value_tensor).all():
        raise RuntimeError(f"Non-finite acquisition value at seed={seed}, iter={iteration}: {acq_value}")

    tolerance = 1e-7
    lower, upper = bounds
    if ((candidate < lower - tolerance) | (candidate > upper + tolerance)).any():
        raise RuntimeError(
            f"Candidate outside [0, 1]^d at seed={seed}, iter={iteration}: {candidate}"
        )


def optimize_acqf_quiet(
    acq_function,
    bounds,
    *,
    q=1,
    num_restarts=ACQ_NUM_RESTARTS,
    raw_samples=ACQ_RAW_SAMPLES,
    options=None,
    seed=None,
    iteration=None,
    model=None,
    **kwargs,
):
    """Optimize an acquisition function while keeping expected SciPy retries quiet."""
    optimizer_options = {**ACQ_OPTIONS, **(options or {})}
    result, warning_records = run_with_warning_capture(
        lambda: optimize_acqf(
            acq_function=acq_function,
            bounds=bounds,
            q=q,
            num_restarts=num_restarts,
            raw_samples=raw_samples,
            options=optimizer_options,
            **kwargs,
        )
    )
    candidate, acq_value = result
    validate_acquisition_result(candidate, acq_value, bounds, seed=seed, iteration=iteration)
    report_warnings(
        "acquisition_optimization",
        warning_records,
        seed=seed,
        iteration=iteration,
        model=model,
        acq_value=acq_value,
    )
    return candidate, acq_value
