"""BEACON's Thompson-sampling novelty acquisition function.

For a posterior sample ``g ~ f | D`` and an archive of previously observed
(or denoised) outcomes ``{y_i}``, BEACON scores a candidate ``x`` by its
distance to the ``k`` nearest archived outcomes (paper, Eqs. 6-7):

.. math::  \\alpha(x) = \\sum_{i=1}^{k} \\mathrm{dist}(g(x), y_{(i)}).

The paper writes the average (``1/k`` times this sum); the original code and
this package use the sum, which has the same maximizer.
"""

from __future__ import annotations

import warnings
from typing import Optional

import torch
from botorch.acquisition import AcquisitionFunction
from botorch.utils.transforms import t_batch_mode_transform

from beacon.models import is_fully_bayesian, submodels
from beacon.thompson_sampling import PathwiseThompsonSampler, pathwise_supported

SAAS_SAMPLE_MODES = ("random", "mean")


def knn_novelty(samples: torch.Tensor, archive: torch.Tensor, k: int) -> torch.Tensor:
    """Sum of Euclidean distances from each sample to its ``k`` nearest archive points.

    Args:
        samples: ``(..., m)`` predicted outcomes.
        archive: ``(N, m)`` archived outcomes.
        k: Number of neighbours (clipped to ``N``).

    Returns:
        ``(...)`` novelty scores (differentiable w.r.t. ``samples``).
    """
    if archive.dim() == 1:
        archive = archive.unsqueeze(-1)
    if samples.shape[-1] != archive.shape[-1]:
        raise ValueError(
            f"Sample outcome dimension {samples.shape[-1]} != archive dimension {archive.shape[-1]}"
        )
    if archive.shape[0] == 0:
        raise ValueError("The archive of observed outcomes is empty.")
    batch_shape = samples.shape[:-1]
    flat = samples.reshape(-1, samples.shape[-1])
    archive = archive.to(dtype=flat.dtype, device=flat.device)
    dist = torch.cdist(flat, archive)
    k_eff = min(int(k), archive.shape[0])
    nearest = torch.topk(dist, k_eff, dim=-1, largest=False).values
    return nearest.sum(dim=-1).reshape(batch_shape)


class BeaconAcquisition(AcquisitionFunction):
    """Continuous-domain BEACON acquisition built on pathwise Thompson samples.

    One differentiable sample path is drawn per outcome (per sub-model of a
    ``ModelListGP``) when the acquisition is constructed and is held fixed while
    the acquisition is optimized, as in Algorithm 1 of the paper.

    Args:
        model: ``ModelListGP`` with one single-output GP per outcome (exact or
            fully Bayesian).
        archive: ``(N, m)`` outcomes to measure novelty against.
        k: Number of nearest neighbours.
        num_bases: Random Fourier features per sample path.
    """

    def __init__(self, model, archive: torch.Tensor, k: int = 10, num_bases: int = 1024):
        super().__init__(model=model)
        self.k = int(k)
        self.archive = archive if archive.dim() == 2 else archive.unsqueeze(-1)
        self.samplers = [PathwiseThompsonSampler(sub, num_of_bases=num_bases) for sub in submodels(model)]

    def sample_outcomes(self, X: torch.Tensor) -> torch.Tensor:
        """Evaluate the fixed sample path: ``(..., q, d) -> (..., q, m)``."""
        return torch.stack([s.query_sample(X) for s in self.samplers], dim=-1)

    @t_batch_mode_transform(expected_q=1)
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        samples = self.sample_outcomes(X).squeeze(-2)  # b x m
        return knn_novelty(samples, self.archive, self.k)


def draw_discrete_samples(
    model,
    candidates: torch.Tensor,
    *,
    method: str = "exact",
    num_bases: int = 1024,
    saas_sample_mode: str = "random",
    chunk_size: int = 8192,
) -> torch.Tensor:
    """Draw one Thompson sample of every outcome at a finite candidate set.

    Args:
        model: ``ModelListGP`` (one GP per outcome) or a single multi-output
            model such as a ``HigherOrderGP``.
        candidates: ``(N, d)`` candidate inputs.
        method: ``"exact"`` draws a joint sample from the exact posterior
            (cost ``O(N^3)``); ``"rff"`` evaluates a pathwise sample (linear in
            ``N``; requires an RBF/Matern kernel).
        num_bases: Random Fourier features for ``"rff"``.
        saas_sample_mode: For fully Bayesian (SAAS) sub-models with ``"exact"``:
            ``"random"`` samples from one randomly chosen MCMC hyperparameter
            sample (a proper Thompson sample of the marginal posterior);
            ``"mean"`` averages independent samples over all MCMC samples
            (the behaviour of the original LogD BEACON-SAAS script).
        chunk_size: Candidates evaluated per batch with ``"rff"``.

    Returns:
        ``(N, m)`` sampled outcomes.
    """
    if method not in ("exact", "rff"):
        raise ValueError(f"Unknown sampling method '{method}' (use 'exact' or 'rff').")
    if saas_sample_mode not in SAAS_SAMPLE_MODES:
        raise ValueError(f"saas_sample_mode must be one of {SAAS_SAMPLE_MODES}")

    outputs = []
    with torch.no_grad():
        for sub in submodels(model):
            if method == "rff":
                sampler = PathwiseThompsonSampler(sub, num_of_bases=num_bases)
                values = torch.cat(
                    [sampler.query_sample(chunk.unsqueeze(-2)).squeeze(-1) for chunk in candidates.split(chunk_size)]
                )
                outputs.append(values.unsqueeze(-1))
                continue

            sample = sub.posterior(candidates).rsample(torch.Size([1]))[0]
            if is_fully_bayesian(sub):  # (n_mcmc, N, 1)
                if saas_sample_mode == "mean":
                    sample = sample.mean(dim=0)
                else:
                    sample = sample[int(torch.randint(sample.shape[0], (1,)).item())]
            outputs.append(sample.reshape(candidates.shape[0], -1))
    return torch.cat(outputs, dim=-1)


def resolve_discrete_method(method: str, n_candidates: int, max_exact: int, model=None) -> str:
    """Pick ``exact`` or ``rff`` for ``method='auto'``.

    Exact joint sampling is used for up to ``max_exact`` candidates, and the
    linear-cost pathwise sampler beyond that (if the kernel supports it).
    """
    if method != "auto":
        if method == "rff" and model is not None and not pathwise_supported(model):
            raise ValueError(
                "discrete_sampler='rff' needs an RBF or Matern kernel; use 'exact' or 'auto' "
                "for other kernels (e.g. Tanimoto)."
            )
        return method
    if n_candidates <= max_exact:
        return "exact"
    if model is not None and not pathwise_supported(model):
        warnings.warn(
            f"The surrogate's kernel has no pathwise sampler, so exact joint sampling is used on "
            f"{n_candidates} candidates (memory ~ {8 * n_candidates**2 / 1e9:.1f} GB per outcome).",
            RuntimeWarning,
        )
        return "exact"
    return "rff"


def candidate_scores(
    samples: torch.Tensor, archive: torch.Tensor, k: int, chunk_size: Optional[int] = 16384
) -> torch.Tensor:
    """Novelty of sampled outcomes ``(N, m)`` w.r.t. ``archive`` (chunked over ``N``)."""
    if chunk_size is None or samples.shape[0] <= chunk_size:
        return knn_novelty(samples, archive, k)
    return torch.cat([knn_novelty(chunk, archive, k) for chunk in samples.split(chunk_size)])
