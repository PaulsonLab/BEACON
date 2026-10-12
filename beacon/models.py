"""GP surrogate construction and fitting.

BEACON models each outcome with an independent GP (the block-diagonal
multi-output kernel used in the paper) and wraps them in a ``ModelListGP``.

Kernels and likelihoods are always constructed *explicitly*. BoTorch changed
the defaults of ``SingleTaskGP`` in v0.12 (Matern-5/2 + Gamma priors became
RBF + dimension-scaled LogNormal priors), so relying on the defaults would
silently change the paper experiments depending on the installed BoTorch
version. ``kernel="matern"`` + ``likelihood="gamma"`` reproduce the
BoTorch 0.10 defaults used for the paper's discrete material benchmarks.
"""

from __future__ import annotations

import inspect
import logging
import warnings
from typing import Sequence

import torch
from botorch.fit import fit_gpytorch_mll
from botorch.models import ModelListGP, SingleTaskGP
from botorch.models.transforms.outcome import Standardize
from botorch.models.utils.gpytorch_modules import (
    get_gaussian_likelihood_with_gamma_prior,
    get_matern_kernel_with_gamma_prior,
)
from gpytorch.constraints import Interval
from gpytorch.kernels import Kernel, RBFKernel, ScaleKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.means import ConstantMean
from gpytorch.mlls import ExactMarginalLogLikelihood

from beacon.optimization import report_warnings, run_with_warning_capture

logger = logging.getLogger(__name__)

KERNELS = ("rbf", "matern", "tanimoto")
LIKELIHOODS = ("gamma", "plain", "noiseless")


class TanimotoKernel(Kernel):
    """Tanimoto (Jaccard) similarity for fingerprints and other non-negative features.

    ``k(x, x') = (<x, x'> + eps) / (eps + ||x||^2 + ||x'||^2 - <x, x'>)``
    with ``eps = 1e-6`` (so two all-zero vectors have similarity 1). It is a
    valid (positive semi-definite) kernel for non-negative inputs and has no
    hyperparameters; wrap it in a ``ScaleKernel`` for an output scale.
    Identical to GAUCHE's ``TanimotoKernel`` used in the paper's ESOL
    experiment, without the dependency.
    """

    is_stationary = False
    has_lengthscale = False
    eps = 1e-6

    def forward(self, x1, x2, diag=False, **params):
        if diag:
            return torch.ones(x1.shape[:-1], dtype=x1.dtype, device=x1.device)
        cross = x1 @ x2.transpose(-2, -1)
        n1 = x1.pow(2).sum(-1, keepdim=True)
        n2 = x2.pow(2).sum(-1, keepdim=True).transpose(-2, -1)
        return ((cross + self.eps) / (self.eps + n1 + n2 - cross)).clamp_min(0)


def make_kernel(kernel: str, dim: int) -> Kernel:
    """Build a covariance module by name (``rbf``, ``matern``, ``tanimoto``)."""
    kernel = kernel.lower()
    if kernel == "rbf":
        # Used by all continuous benchmarks and the MOF / oil multi-outcome studies.
        return ScaleKernel(RBFKernel(ard_num_dims=dim))
    if kernel == "matern":
        # BoTorch <= 0.11 SingleTaskGP default (Matern-5/2 with Gamma priors).
        return get_matern_kernel_with_gamma_prior(ard_num_dims=dim)
    if kernel == "tanimoto":
        return ScaleKernel(TanimotoKernel())
    raise ValueError(f"Unknown kernel '{kernel}'. Choose from {KERNELS}.")


def make_likelihood(likelihood: str) -> GaussianLikelihood:
    """Build a Gaussian likelihood by name.

    * ``gamma``: BoTorch <= 0.11 default (Gamma(1.1, 0.05) prior, noise >= 1e-4).
    * ``plain``: GPyTorch default, no prior (noise >= 1e-4). Used when
      observation noise is expected (noisy Ackley, ESOL).
    * ``noiseless``: noise constrained to [1e-6, 1e-4] (interpolating GP).
    """
    likelihood = likelihood.lower()
    if likelihood == "gamma":
        return get_gaussian_likelihood_with_gamma_prior()
    if likelihood == "plain":
        return GaussianLikelihood()
    if likelihood == "noiseless":
        return GaussianLikelihood(noise_constraint=Interval(1e-6, 1e-4))
    raise ValueError(f"Unknown likelihood '{likelihood}'. Choose from {LIKELIHOODS}.")


def build_single_output_gp(
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    kernel: str = "rbf",
    likelihood: str = "gamma",
) -> SingleTaskGP:
    """Single-output exact GP with standardized outcomes."""
    if train_y.dim() == 1:
        train_y = train_y.unsqueeze(-1)
    if kernel.lower() == "tanimoto" and bool((train_x < 0).any()):
        raise ValueError(
            "The Tanimoto kernel needs non-negative inputs (e.g. binary or count fingerprints, "
            "or features scaled to [0, 1])."
        )
    model = SingleTaskGP(
        train_x,
        train_y,
        covar_module=make_kernel(kernel, train_x.shape[-1]),
        likelihood=make_likelihood(likelihood),
        mean_module=ConstantMean(),
        outcome_transform=Standardize(m=1),
    )
    return model.to(train_x)


def fit_exact_gp(model, *, seed=None, iteration=None, raise_on_failure: bool = False) -> None:
    """Maximize the marginal likelihood (L-BFGS-B with BoTorch's retries).

    If every retry fails, a warning is issued and the model keeps its last
    hyperparameters (the run continues) unless ``raise_on_failure`` is set.
    """
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    try:
        _, warning_records = run_with_warning_capture(lambda: fit_gpytorch_mll(mll))
        report_warnings("gp_fit", warning_records, seed=seed, iteration=iteration, model=model)
    except Exception as exc:  # pragma: no cover - depends on numerical failures
        if raise_on_failure:
            raise RuntimeError(f"GP fitting failed at seed={seed}, iteration={iteration}") from exc
        warnings.warn(
            f"GP fitting failed at seed={seed}, iteration={iteration} ({exc}); "
            "continuing with the current hyperparameters.",
            RuntimeWarning,
        )
    model.eval()


def build_independent_gps(
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    kernel: str = "rbf",
    likelihood: str = "gamma",
    *,
    seed=None,
    iteration=None,
) -> ModelListGP:
    """Fit one independent GP per outcome column and return a ``ModelListGP``."""
    if train_y.dim() == 1:
        train_y = train_y.unsqueeze(-1)
    models = []
    for j in range(train_y.shape[-1]):
        gp = build_single_output_gp(train_x, train_y[:, j : j + 1], kernel, likelihood)
        fit_exact_gp(gp, seed=seed, iteration=iteration)
        models.append(gp)
    return ModelListGP(*models).eval()


def build_saas_gps(
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    *,
    warmup_steps: int = 128,
    num_samples: int = 64,
    thinning: int = 16,
    max_tree_depth: int = 6,
) -> ModelListGP:
    """Fit one SAAS fully Bayesian GP per outcome with NUTS.

    The defaults (128 warm-up steps, 64 samples, thinning 16 -> 4 retained
    MCMC samples) are those of the paper's LogD BEACON-SAAS experiment.
    """
    try:
        from botorch.fit import fit_fully_bayesian_model_nuts
        from botorch.models.fully_bayesian import SaasFullyBayesianSingleTaskGP
    except ImportError as exc:  # pragma: no cover
        raise ImportError("BEACON-SAAS requires BoTorch's fully Bayesian models.") from exc

    if train_y.dim() == 1:
        train_y = train_y.unsqueeze(-1)
    fit_kwargs = dict(
        warmup_steps=warmup_steps,
        num_samples=num_samples,
        thinning=thinning,
        max_tree_depth=max_tree_depth,
        disable_progbar=True,
    )
    accepted = inspect.signature(fit_fully_bayesian_model_nuts).parameters
    models = []
    for j in range(train_y.shape[-1]):
        try:
            gp = SaasFullyBayesianSingleTaskGP(
                train_X=train_x, train_Y=train_y[:, j : j + 1], outcome_transform=Standardize(m=1)
            )
        except ImportError as exc:
            raise ImportError(
                "BEACON-SAAS needs the fully Bayesian extras of BoTorch. Install them with "
                "`pip install \"botorch[fully_bayesian]\"` (BoTorch >= 0.12: JAX + NumPyro) "
                "or `pip install pyro-ppl` (BoTorch <= 0.11)."
            ) from exc
        kwargs = {k: v for k, v in fit_kwargs.items() if k in accepted}
        if "seed" in accepted:
            # Newer BoTorch seeds NUTS with 0 by default; derive the seed from torch's
            # RNG so chains differ across iterations but remain reproducible.
            kwargs["seed"] = int(torch.randint(0, 2**31 - 1, (1,)).item())
        fit_fully_bayesian_model_nuts(gp, **kwargs)
        models.append(gp)
    return ModelListGP(*models).eval()


def build_hogp(
    train_x: torch.Tensor,
    train_y_flat: torch.Tensor,
    output_shape: Sequence[int],
    *,
    step_limit: int = 1000,
    lr: float = 0.05,
):
    """Higher-order GP for tensor-valued outcomes (MNIST images in the paper)."""
    from functools import partial

    from botorch.models import HigherOrderGP
    from botorch.models.higher_order_gp import FlattenedStandardize
    from botorch.optim.fit import fit_gpytorch_mll_torch
    from linear_operator.settings import _fast_solves

    train_y = train_y_flat.reshape(train_y_flat.shape[0], *output_shape)
    model = HigherOrderGP(
        train_x,
        train_y,
        outcome_transform=FlattenedStandardize(train_y.shape[1:]),
        latent_init="gp",
    )
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    with _fast_solves(True):
        fit_gpytorch_mll_torch(mll, step_limit=step_limit, optimizer=partial(torch.optim.Adam, lr=lr))
    return model.eval()


def is_fully_bayesian(model) -> bool:
    """Whether ``model`` carries a batch of MCMC hyperparameter samples."""
    return bool(getattr(model, "_is_fully_bayesian", False))


def submodels(model) -> list:
    """Sub-models of a ``ModelListGP`` (or ``[model]`` for any other model)."""
    return list(model.models) if isinstance(model, ModelListGP) else [model]


def posterior_mean(model, X: torch.Tensor) -> torch.Tensor:
    """Posterior mean of ``model`` at ``X`` as an ``(n, m)`` tensor.

    Fully Bayesian sub-models are averaged over their MCMC samples, i.e. this
    is the mean of the hyperparameter-marginalized posterior. Tensor-valued
    outcomes (HOGP) are flattened to ``m = prod(output_shape)`` columns.
    """
    means = []
    with torch.no_grad():
        for sub in submodels(model):
            mean = sub.posterior(X).mean
            if is_fully_bayesian(sub):
                mean = mean.mean(dim=0)  # (n_mcmc, n, 1) -> (n, 1)
            means.append(mean.reshape(X.shape[0], -1))
    return torch.cat(means, dim=-1).detach()
