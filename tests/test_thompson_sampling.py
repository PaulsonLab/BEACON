"""Correctness checks for the pathwise (decoupled) Thompson sampler.

The key property: averaged over many draws, sample paths must reproduce the
exact GP posterior mean and standard deviation, both near the data and far
from it (where the posterior reverts to the prior). The original
implementation failed the far-field check by a factor sqrt(outputscale).
"""

import math

import pytest
import torch
from botorch.models import SingleTaskGP
from botorch.models.transforms.outcome import Standardize
from gpytorch.kernels import MaternKernel, RBFKernel, ScaleKernel

from beacon.models import fit_exact_gp
from beacon.thompson_sampling import EfficientThompsonSampler, PathwiseThompsonSampler
from conftest import toy_data


def fitted_gp(covar_module, device="cpu"):
    torch.manual_seed(0)
    X, Y = toy_data(n=15, device=device)
    model = SingleTaskGP(X, Y, covar_module=covar_module, outcome_transform=Standardize(m=1)).to(X)
    fit_exact_gp(model)
    return model, X


def empirical_moments(model, Xq, n_draws=1500):
    sampler = PathwiseThompsonSampler(model, num_of_bases=1024)
    with torch.no_grad():
        draws = torch.stack([sampler.resample().query_sample(Xq.unsqueeze(-2)).squeeze(-1) for _ in range(n_draws)])
        post = model.posterior(Xq)
        mu, sd = post.mean.squeeze(-1), post.variance.sqrt().squeeze(-1)
    return draws.mean(0), draws.std(0), mu, sd


KERNELS = {
    "scaled-rbf": lambda: ScaleKernel(RBFKernel(ard_num_dims=2)),
    "scaled-matern52": lambda: ScaleKernel(MaternKernel(nu=2.5, ard_num_dims=2)),
    "matern32-unscaled": lambda: MaternKernel(nu=1.5, ard_num_dims=2),
}


@pytest.mark.parametrize("kernel", sorted(KERNELS))
def test_sample_moments_match_exact_posterior(kernel):
    model, _ = fitted_gp(KERNELS[kernel]())
    g = torch.Generator().manual_seed(1)
    near = torch.rand(4, 2, generator=g, dtype=torch.double)
    far = torch.rand(3, 2, generator=g, dtype=torch.double) + 4.0  # prior-dominated region
    mean, std, mu, sd = empirical_moments(model, torch.cat([near, far]))
    # Monte-Carlo error of the mean is ~ sd / sqrt(1500) ~ 0.026 sd.
    assert torch.all((mean - mu).abs() <= 0.15 * sd), (mean, mu, sd)
    assert torch.all((std / sd - 1).abs() <= 0.12), (std, sd)


def test_kernel_matches_gpytorch():
    model, X = fitted_gp(ScaleKernel(MaternKernel(nu=2.5, ard_num_dims=2)))
    sampler = PathwiseThompsonSampler(model)
    with torch.no_grad():
        assert torch.allclose(sampler.kernel(X, X), model.covar_module(X, X).to_dense(), atol=1e-9)


def test_sample_is_fixed_until_resampled():
    model, _ = fitted_gp(ScaleKernel(RBFKernel(ard_num_dims=2)))
    sampler = PathwiseThompsonSampler(model)
    x = torch.tensor([[[0.25, 0.75]]], dtype=torch.double)
    first = sampler.query_sample(x).detach().clone()
    assert torch.equal(first, sampler.query_sample(x).detach())
    sampler.calculate_V()  # new pseudo-noise -> new sample
    assert not torch.equal(first, sampler.query_sample(x).detach())


def test_sample_is_differentiable_and_shapes():
    model, _ = fitted_gp(ScaleKernel(RBFKernel(ard_num_dims=2)))
    sampler = EfficientThompsonSampler(model)  # backward-compatible alias
    x = torch.rand(5, 1, 2, dtype=torch.double, requires_grad=True)
    values = sampler.query_sample(x)
    assert values.shape == (5, 1)
    values.sum().backward()
    assert torch.isfinite(x.grad).all() and x.grad.abs().sum() > 0


def test_sample_interpolates_data_for_noiseless_gp():
    model, X = fitted_gp(ScaleKernel(RBFKernel(ard_num_dims=2)))
    with torch.no_grad():
        y = model.posterior(X).mean.squeeze(-1)
        sample = PathwiseThompsonSampler(model).query_sample(X.unsqueeze(-2)).squeeze(-1)
        sd = model.posterior(X).variance.sqrt().squeeze(-1)
    assert torch.all((sample - y).abs() <= 6 * sd + 1e-3)


def test_dtype_and_device_preserved(device):
    model, _ = fitted_gp(ScaleKernel(RBFKernel(ard_num_dims=2)), device=device)
    sampler = PathwiseThompsonSampler(model)
    out = sampler.query_sample(torch.rand(3, 1, 2, dtype=torch.double, device=device))
    for t in (sampler.thetas, sampler.weights, sampler.Phi, sampler.V, out):
        assert t.dtype == torch.double and t.device.type == device.type


def test_unsupported_kernel_raises():
    from beacon.models import TanimotoKernel

    torch.manual_seed(0)
    X = (torch.rand(6, 8) > 0.5).double()
    model = SingleTaskGP(X, torch.rand(6, 1, dtype=torch.double), covar_module=ScaleKernel(TanimotoKernel()))
    with pytest.raises(NotImplementedError):
        PathwiseThompsonSampler(model)


def test_matern_spectral_frequencies_have_t_tails():
    """Matern-1/2 frequencies follow a Cauchy law: many more outliers than RBF."""
    model, _ = fitted_gp(MaternKernel(nu=0.5, ard_num_dims=2))
    sampler = PathwiseThompsonSampler(model, num_of_bases=20000)
    z = (sampler.thetas * sampler.lengthscale).abs()
    assert (z > 5).float().mean() > 0.05  # Gaussian would give ~6e-7
    assert math.isclose(sampler.nu, 0.5)
