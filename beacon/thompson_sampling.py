"""Differentiable approximate Thompson samples from GP posteriors.

Implements the decoupled ("pathwise") sampling rule of Wilson et al. (2020),
`Efficiently Sampling Functions from Gaussian Process Posteriors
<https://arxiv.org/abs/2002.09309>`_:

.. math::

    g(x) = m(x) + \\phi(x)^\\top w
           + k(x, X)\\,(K_{XX} + \\Sigma)^{-1}\\,(y - m(X) - \\Phi w - \\varepsilon),

where :math:`\\phi` are random Fourier features of the (stationary) kernel,
:math:`w \\sim N(0, I)`, :math:`\\varepsilon \\sim N(0, \\Sigma)` and
:math:`\\Sigma` is the (homoskedastic or fixed) observation-noise covariance.
All randomness is drawn once per sample, so ``g`` is a deterministic,
differentiable function that can be maximized with gradient methods.

The sampler works in the GP's *transformed* outcome space (e.g. after
``Standardize``) and maps samples back to the original outcome scale with the
model's ``outcome_transform``.

Supported models: single-output ``SingleTaskGP``-style exact GPs with a
constant or zero mean and an RBF or Matern (nu = 0.5, 1.5, 2.5) kernel,
optionally wrapped in a ``ScaleKernel``. Fully Bayesian models (e.g.
``SaasFullyBayesianSingleTaskGP``) are supported by selecting one MCMC
hyperparameter sample (``mcmc_index``; random by default), which yields a
valid Thompson sample from the hyperparameter-marginalized posterior.

Fixes relative to the original research implementation:

* Fourier features are scaled by ``sqrt(outputscale)`` (previously
  ``outputscale``), so the prior sample has the kernel's variance.
* The pseudo-observation noise is drawn with standard deviation
  ``sqrt(noise)`` (previously the noise *variance* was used as the std).
* The GP's learned constant prior mean is included.
* Linear solves use a Cholesky factorization instead of an explicit inverse.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
from gpytorch.kernels import MaternKernel, RBFKernel, ScaleKernel
from gpytorch.means import ConstantMean, ZeroMean


class PathwiseThompsonSampler:
    """Draw differentiable approximate posterior samples of a single-output GP.

    Args:
        model: A fitted single-output exact GP (see module docstring).
        num_of_multistarts: Number of random restarts used by
            :meth:`generate_candidates` (only relevant for that helper).
        num_of_bases: Number of random Fourier features ``L``.
        num_of_samples: Number of independent posterior samples ``S``. When
            ``S > 1`` the ``q`` dimension of the query must equal ``S`` (sample
            ``j`` is evaluated at point ``j``) or be 1.
        mcmc_index: For fully Bayesian models, which MCMC hyperparameter sample
            to use. ``None`` selects one uniformly at random.

    Usage::

        sampler = PathwiseThompsonSampler(model)
        sampler.resample()              # draw a new sample path
        values = sampler.query_sample(X)  # X: (..., q, d) -> (..., q)
    """

    def __init__(
        self,
        model,
        num_of_multistarts: int = 1,
        num_of_bases: int = 1024,
        num_of_samples: int = 1,
        mcmc_index: Optional[int] = None,
    ):
        self.model = model
        self.num_of_multistarts = num_of_multistarts
        self.num_of_bases = num_of_bases
        self.num_of_samples = num_of_samples
        self._extract_hyperparameters(mcmc_index)

        # Optimization settings for generate_candidates (kept from the original).
        self.learning_rate = 0.01
        self.num_of_epochs = 10 * self.x_dim

        with torch.no_grad():
            K = self.kernel(self.train_x, self.train_x)
            noise = self.noise.expand(self.num_of_train_inputs)
            self._chol = _robust_cholesky(K + torch.diag_embed(noise))

        self.create_fourier_bases()
        self.calculate_phi()
        self.create_sample()
        self.calculate_V()

    # ------------------------------------------------------------------
    # Hyperparameter extraction
    # ------------------------------------------------------------------
    def _extract_hyperparameters(self, mcmc_index: Optional[int]) -> None:
        model = self.model
        if getattr(model, "num_outputs", 1) != 1:
            raise ValueError(
                "PathwiseThompsonSampler expects a single-output GP; build one sampler "
                "per output (e.g. per sub-model of a ModelListGP)."
            )

        covar = model.covar_module
        if isinstance(covar, ScaleKernel):
            outputscale = covar.outputscale.detach()
            base = covar.base_kernel
        else:
            outputscale = None
            base = covar

        if isinstance(base, MaternKernel):
            self.nu = float(base.nu)
            if self.nu not in (0.5, 1.5, 2.5):
                raise NotImplementedError(f"Matern nu={self.nu} is not supported.")
        elif isinstance(base, RBFKernel):
            self.nu = math.inf
        else:
            raise NotImplementedError(
                f"Pathwise sampling needs an RBF or Matern kernel, got {type(base).__name__}. "
                "Use exact joint sampling over a discrete candidate set instead."
            )

        lengthscale = base.lengthscale.detach()  # (*batch, 1, d)
        train_x = model.train_inputs[0].detach()  # (*batch, n, d)
        train_y = model.train_targets.detach()  # (*batch, n), transformed scale
        noise = model.likelihood.noise.detach()  # (*batch, 1) or (*batch, n)
        mean_module = model.mean_module
        if isinstance(mean_module, ConstantMean):
            mean_constant = mean_module.constant.detach()
        elif isinstance(mean_module, ZeroMean):
            mean_constant = torch.zeros((), dtype=train_x.dtype, device=train_x.device)
        else:
            raise NotImplementedError(
                f"Pathwise sampling supports constant or zero means, got {type(mean_module).__name__}."
            )

        batch_shape = lengthscale.shape[:-2]
        if len(batch_shape) > 1:
            raise NotImplementedError("Only one batch dimension (MCMC samples) is supported.")
        if len(batch_shape) == 1:
            num_mcmc = batch_shape[0]
            if mcmc_index is None:
                mcmc_index = int(torch.randint(num_mcmc, (1,)).item())
            if not 0 <= mcmc_index < num_mcmc:
                raise IndexError(f"mcmc_index={mcmc_index} outside [0, {num_mcmc}).")

            def pick(t):
                return t[mcmc_index] if t.dim() > 0 and t.shape[0] == num_mcmc else t

            lengthscale = lengthscale[mcmc_index]
            outputscale = None if outputscale is None else pick(outputscale)
            train_x = train_x[mcmc_index] if train_x.dim() == 3 else train_x
            train_y = train_y[mcmc_index] if train_y.dim() == 2 else train_y
            noise = pick(noise)
            mean_constant = pick(mean_constant)
        self.mcmc_index = mcmc_index

        self.train_x = train_x
        self.dtype = train_x.dtype
        self.device = train_x.device
        self.num_of_train_inputs, self.x_dim = train_x.shape
        self.scaled_train_y = train_y.reshape(-1).to(self.dtype)
        self.lengthscale = lengthscale.reshape(1, -1).to(self.dtype)
        self.outputscale = (
            torch.ones((), dtype=self.dtype, device=self.device)
            if outputscale is None
            else outputscale.reshape(()).to(self.dtype)
        )
        self.noise = noise.reshape(-1).to(self.dtype)
        if self.noise.numel() not in (1, self.num_of_train_inputs):
            raise ValueError("Observation noise must be scalar or have one entry per training point.")
        self.mean_constant = mean_constant.reshape(()).to(self.dtype)

    # ------------------------------------------------------------------
    # Kernel and Fourier features
    # ------------------------------------------------------------------
    def kernel(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        """Evaluate the (scaled) stationary kernel, broadcasting batch dims."""
        z1 = x1 / self.lengthscale
        z2 = x2 / self.lengthscale
        sq = (z1.unsqueeze(-2) - z2.unsqueeze(-3)).pow(2).sum(-1)
        if math.isinf(self.nu):
            k = torch.exp(-0.5 * sq)
        else:
            r = sq.clamp_min(1e-30).sqrt()
            if self.nu == 0.5:
                k = torch.exp(-r)
            elif self.nu == 1.5:
                s3r = math.sqrt(3.0) * r
                k = (1.0 + s3r) * torch.exp(-s3r)
            else:
                s5r = math.sqrt(5.0) * r
                k = (1.0 + s5r + (5.0 / 3.0) * sq) * torch.exp(-s5r)
        return self.outputscale * k

    def create_fourier_bases(self) -> None:
        """Sample spectral frequencies ``thetas`` (L x d) and phases ``biases`` (L)."""
        z = torch.randn(self.num_of_bases, self.x_dim, dtype=self.dtype, device=self.device)
        if not math.isinf(self.nu):
            # Matern spectral density = multivariate Student-t with 2*nu dof.
            chi2 = torch.distributions.Chi2(
                torch.tensor(2.0 * self.nu, dtype=self.dtype, device=self.device)
            ).sample((self.num_of_bases, 1))
            z = z * torch.sqrt(2.0 * self.nu / chi2)
        self.thetas = z / self.lengthscale
        self.biases = torch.rand(self.num_of_bases, dtype=self.dtype, device=self.device) * 2 * math.pi

    def calculate_fourier_features(self, x: torch.Tensor) -> torch.Tensor:
        """Random Fourier features ``phi(x)`` with ``E[phi(x) phi(x')] = k(x, x')``."""
        scale = torch.sqrt(2.0 * self.outputscale / self.num_of_bases)
        return scale * torch.cos(x @ self.thetas.T + self.biases)

    def calculate_phi(self) -> None:
        """Fourier features at the training inputs (n x L)."""
        self.Phi = self.calculate_fourier_features(self.train_x)

    # ------------------------------------------------------------------
    # Sample construction
    # ------------------------------------------------------------------
    def create_sample(self) -> None:
        """Draw prior weights ``w`` (S x L) for a new sample path."""
        self.weights = torch.randn(
            self.num_of_samples, self.num_of_bases, dtype=self.dtype, device=self.device
        )

    def calculate_V(self) -> None:
        """Pre-compute the pathwise-update weights ``V`` (n x S)."""
        prior_at_train = self.Phi @ self.weights.T  # n x S
        eps = torch.randn_like(prior_at_train) * self.noise.sqrt().unsqueeze(-1)
        residual = (self.scaled_train_y - self.mean_constant).unsqueeze(-1) - prior_at_train - eps
        self.V = torch.cholesky_solve(residual, self._chol)

    def resample(self) -> "PathwiseThompsonSampler":
        """Draw a fresh sample path (new Fourier bases, weights and noise)."""
        self.create_fourier_bases()
        self.calculate_phi()
        self.create_sample()
        self.calculate_V()
        return self

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------
    def _as_tensor(self, x) -> torch.Tensor:
        if not torch.is_tensor(x):
            return torch.as_tensor(x, dtype=self.dtype, device=self.device)
        return x.to(dtype=self.dtype, device=self.device)

    def sample_prior(self, x) -> torch.Tensor:
        """Prior part ``phi(x) w`` of the sample, shape (..., q)."""
        x = self._as_tensor(x)
        return (self.calculate_fourier_features(x) * self.weights).sum(-1)

    def posterior_update(self, x) -> torch.Tensor:
        """Data-dependent part ``k(x, X) V`` of the sample, shape (..., q)."""
        x = self._as_tensor(x)
        kxX = self.kernel(x, self.train_x)  # (..., q, n)
        return (kxX * self.V.T).sum(-1)

    def query_sample_scaled(self, x) -> torch.Tensor:
        """Sample value in the GP's transformed outcome space, shape (..., q)."""
        return self.mean_constant + self.sample_prior(x) + self.posterior_update(x)

    def query_sample(self, x) -> torch.Tensor:
        """Sample value on the original outcome scale.

        Args:
            x: Inputs of shape ``(..., q, d)``.

        Returns:
            Tensor of shape ``(..., q)``.
        """
        y_scaled = self.query_sample_scaled(x)
        transform = getattr(self.model, "outcome_transform", None)
        if transform is None:
            return y_scaled
        return transform.untransform(y_scaled.unsqueeze(-1))[0].squeeze(-1)

    __call__ = query_sample

    def generate_candidates(self) -> torch.Tensor:
        """Maximize the sample(s) over ``[0, 1]^d`` with multi-start Adam.

        Returns:
            Tensor of shape ``(num_of_samples, d)``: the best point per sample.
        """
        X = torch.rand(
            self.num_of_multistarts, self.num_of_samples, self.x_dim, dtype=self.dtype, device=self.device
        )
        X.requires_grad_(True)
        optimiser = torch.optim.Adam([X], lr=self.learning_rate)
        for _ in range(self.num_of_epochs):
            optimiser.zero_grad()
            loss = -self.query_sample(X).sum()
            loss.backward()
            optimiser.step()
            with torch.no_grad():
                X.clamp_(0.0, 1.0)
        with torch.no_grad():
            final_evals = self.query_sample(X)
            best_idx = torch.argmax(final_evals, dim=0)
            return X[best_idx, torch.arange(self.num_of_samples), :].detach()


# Backward-compatible name used by the original experiment scripts.
EfficientThompsonSampler = PathwiseThompsonSampler


def pathwise_supported(model) -> bool:
    """Whether :class:`PathwiseThompsonSampler` can sample every outcome of ``model``.

    Requires single-output sub-models with a constant/zero mean and an RBF or
    Matern (nu in {0.5, 1.5, 2.5}) kernel, optionally inside a ``ScaleKernel``.
    Non-stationary kernels such as Tanimoto have no random-Fourier-feature
    expansion and need exact joint sampling instead.
    """
    from botorch.models import ModelListGP

    subs = list(model.models) if isinstance(model, ModelListGP) else [model]
    for sub in subs:
        if getattr(sub, "num_outputs", 1) != 1:
            return False
        if not isinstance(getattr(sub, "mean_module", None), (ConstantMean, ZeroMean)):
            return False
        covar = getattr(sub, "covar_module", None)
        base = covar.base_kernel if isinstance(covar, ScaleKernel) else covar
        if isinstance(base, MaternKernel):
            if float(base.nu) not in (0.5, 1.5, 2.5):
                return False
        elif not isinstance(base, RBFKernel):
            return False
    return True


def _robust_cholesky(A: torch.Tensor, max_tries: int = 6) -> torch.Tensor:
    """Cholesky factor with escalating diagonal jitter for near-singular matrices."""
    L, info = torch.linalg.cholesky_ex(A)
    if not bool(info.any()):
        return L
    jitter = 1e-9 * A.diagonal(dim1=-2, dim2=-1).mean().clamp_min(1e-12)
    eye = torch.eye(A.shape[-1], dtype=A.dtype, device=A.device)
    for _ in range(max_tries):
        L, info = torch.linalg.cholesky_ex(A + jitter * eye)
        if not bool(info.any()):
            return L
        jitter = jitter * 10
    raise RuntimeError("Kernel matrix is not positive definite, even after adding jitter.")
