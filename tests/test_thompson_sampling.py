"""Regression checks for the efficient Thompson sampler."""

import types

import torch
from botorch.models import SingleTaskGP
from botorch.models.transforms.outcome import Standardize
from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.likelihoods import GaussianLikelihood

from beacon.thompson_sampling import EfficientThompsonSampler


def build_sampler():
    torch.manual_seed(123)
    train_x = torch.linspace(0, 1, 6, dtype=torch.double).unsqueeze(-1)
    train_y = torch.sin(train_x * 6.28)
    model = SingleTaskGP(
        train_x,
        train_y,
        covar_module=ScaleKernel(MaternKernel()),
        likelihood=GaussianLikelihood(),
        outcome_transform=Standardize(m=1),
    )
    # The experiment scripts attach these attributes before constructing the sampler.
    model.train_x = train_x
    model.train_y = train_y

    sampler = EfficientThompsonSampler(
        model,
        num_of_multistarts=1,
        num_of_bases=64,
        num_of_samples=1,
    )
    sampler.create_sample()
    sampler.calculate_V()
    return sampler


def test_query_sample_reuses_fixed_v_for_same_sample():
    sampler = build_sampler()
    x = torch.tensor([[[0.25]]])

    first = sampler.query_sample(x).detach().clone()
    second = sampler.query_sample(x).detach().clone()
    assert torch.equal(first, second)

    sampler.calculate_V()
    recalculated = sampler.query_sample(x).detach().clone()
    assert not torch.equal(first, recalculated)


def test_query_sample_computes_posterior_update_once():
    sampler = build_sampler()
    original_posterior_update = sampler.posterior_update
    call_count = 0

    def counted_posterior_update(self, x):
        nonlocal call_count
        call_count += 1
        return original_posterior_update(x)

    sampler.posterior_update = types.MethodType(counted_posterior_update, sampler)
    sampler.query_sample(torch.tensor([[[0.25]]]))

    assert call_count == 1


def test_sampler_preserves_model_dtype():
    sampler = build_sampler()
    x = torch.tensor([[[0.25]]], dtype=torch.double)

    assert sampler.thetas.dtype == torch.double
    assert sampler.weights.dtype == torch.double
    assert sampler.Phi.dtype == torch.double
    assert sampler.V.dtype == torch.double
    assert sampler.query_sample(x).dtype == torch.double
