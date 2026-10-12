"""BEACON, UG-BEACON and BEACON-SAAS.

All three share Algorithm 1 of the paper:

1. fit an independent GP per outcome to the data,
2. draw one Thompson sample ``g`` of the outcome function,
3. pick the input whose sampled outcome ``g(x)`` is farthest (k-NN distance)
   from the archive of observed outcomes.

They differ in a single component:

=============  =========================================================
``beacon``      exact GPs (MAP hyperparameters); optimizes the acquisition
                over a continuous box (L-BFGS-B, multi-start) or scores a
                finite candidate set.
``beacon-saas`` replaces each GP by a SAAS fully Bayesian GP (sparse axis-
                aligned subspace prior, fitted with NUTS) for problems with
                many input dimensions of which few matter.
``ug-beacon``   *user-guided* BEACON: only inputs whose sampled outcome
                falls outside the already-visited (or excluded) behaviour
                regions of a user-supplied partition are eligible.
=============  =========================================================

Use :func:`get_algorithm` (or the ``--algorithm`` CLI flag) to select one.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Optional, Sequence

import torch
from torch.quasirandom import SobolEngine

from beacon.acquisition import (
    SAAS_SAMPLE_MODES,
    BeaconAcquisition,
    candidate_scores,
    draw_discrete_samples,
    resolve_discrete_method,
)
from beacon.metrics import OutcomePartition
from beacon.models import (
    KERNELS,
    LIKELIHOODS,
    build_hogp,
    build_independent_gps,
    build_saas_gps,
    posterior_mean,
)
from beacon.optimization import optimize_acqf_quiet
from beacon.thompson_sampling import pathwise_supported

ARCHIVE_MODES = ("observed", "posterior_mean")
SURROGATES = ("gp", "saas", "hogp")
DISCRETE_SAMPLERS = ("auto", "exact", "rff")


@dataclass
class BeaconConfig:
    """Hyperparameters shared by all BEACON variants.

    Problem presets (see :mod:`beacon.problems`) override some of these to
    match the paper's experimental settings; user overrides take precedence.
    """

    # Novelty acquisition
    k: int = 10
    """Number of nearest neighbours in the novelty score."""
    archive: str = "posterior_mean"
    """Outcomes novelty is measured against: ``observed`` values, or the GP
    ``posterior_mean`` at the sampled inputs (denoised archive; paper Eq. 6)."""

    # Surrogate model
    surrogate: str = "gp"
    """``gp`` (exact GP per outcome), ``saas`` (SAAS fully Bayesian GP per
    outcome) or ``hogp`` (higher-order GP for tensor outcomes)."""
    kernel: str = "rbf"
    """``rbf``, ``matern`` (Matern-5/2 + Gamma priors) or ``tanimoto``
    (Jaccard similarity for fingerprints / non-negative features; ``gp``
    surrogate only, sampled exactly rather than pathwise)."""
    likelihood: str = "gamma"
    """``gamma`` (Gamma noise prior), ``plain`` (no prior) or ``noiseless``."""

    # Thompson sampling
    num_bases: int = 1024
    """Random Fourier features per pathwise sample."""
    discrete_sampler: str = "auto"
    """For candidate sets: ``exact`` joint sampling, pathwise ``rff``, or
    ``auto`` (exact up to ``max_exact_candidates`` candidates)."""
    max_exact_candidates: int = 6000

    # Continuous acquisition optimization (BoTorch optimize_acqf)
    num_restarts: int = 10
    raw_samples: int = 64
    maxiter: int = 200
    batch_limit: int = 5

    # BEACON-SAAS (NUTS)
    warmup_steps: int = 128
    num_samples: int = 64
    thinning: int = 16
    saas_sample_mode: str = "random"
    """``random``: Thompson sample from one random MCMC sample; ``mean``:
    average samples over MCMC draws (original LogD script)."""

    # Pool-based selection on continuous domains
    pool_size: int = 4096
    """Sobol points scored on continuous domains by UG-BEACON, and by BEACON
    when the kernel has no pathwise sampler (e.g. Tanimoto)."""

    # UG-BEACON
    ug_excluded_regions: Sequence[int] = field(default_factory=tuple)
    """Partition regions the user wants to avoid, in addition to visited ones."""

    # Higher-order GP
    hogp_steps: int = 1000
    hogp_lr: float = 0.05

    def __post_init__(self):
        self.validate()

    def validate(self) -> None:
        def check(name, value, options):
            if value not in options:
                raise ValueError(f"{name}='{value}' is invalid; choose from {options}")

        check("archive", self.archive, ARCHIVE_MODES)
        check("surrogate", self.surrogate, SURROGATES)
        check("kernel", self.kernel, KERNELS)
        check("likelihood", self.likelihood, LIKELIHOODS)
        check("discrete_sampler", self.discrete_sampler, DISCRETE_SAMPLERS)
        check("saas_sample_mode", self.saas_sample_mode, SAAS_SAMPLE_MODES)
        if self.k < 1:
            raise ValueError("k must be >= 1")
        if self.pool_size < 1:
            raise ValueError("pool_size must be >= 1")
        if self.kernel == "tanimoto":
            if self.surrogate != "gp":
                raise ValueError(
                    f"kernel='tanimoto' is only available with surrogate='gp' (got '{self.surrogate}'); "
                    "SAAS and HOGP surrogates use their own kernels."
                )
            if self.discrete_sampler == "rff":
                raise ValueError(
                    "kernel='tanimoto' has no random-Fourier-feature sampler; "
                    "use discrete_sampler='exact' or 'auto'."
                )

    @classmethod
    def field_names(cls) -> set:
        return {f.name for f in dataclasses.fields(cls)}

    def replace(self, **overrides) -> "BeaconConfig":
        unknown = set(overrides) - self.field_names()
        if unknown:
            raise ValueError(f"Unknown BEACON option(s): {sorted(unknown)}")
        return dataclasses.replace(self, **overrides)

    def to_dict(self) -> dict:
        d = dataclasses.asdict(self)
        d["ug_excluded_regions"] = list(d["ug_excluded_regions"])
        return d


class BEACON:
    """Standard BEACON (paper Algorithm 1).

    Args:
        config: A :class:`BeaconConfig`; keyword overrides are applied on top.
        output_shape: Shape of tensor-valued outcomes (HOGP surrogate only).
    """

    name = "beacon"
    surrogate_override: Optional[str] = None

    def __init__(self, config: Optional[BeaconConfig] = None, *, output_shape=None, **overrides):
        config = config or BeaconConfig()
        if self.surrogate_override is not None:
            overrides.setdefault("surrogate", self.surrogate_override)
        self.config = config.replace(**overrides) if overrides else config
        self.output_shape = output_shape

    # ------------------------------------------------------------------
    def fit(self, train_x, train_y, *, seed=None, iteration=None):
        """Fit the surrogate to ``(train_x, train_y)`` and return it."""
        cfg = self.config
        if cfg.surrogate == "gp":
            return build_independent_gps(train_x, train_y, cfg.kernel, cfg.likelihood, seed=seed, iteration=iteration)
        if cfg.surrogate == "saas":
            return build_saas_gps(
                train_x, train_y, warmup_steps=cfg.warmup_steps, num_samples=cfg.num_samples, thinning=cfg.thinning
            )
        if cfg.surrogate == "hogp":
            if self.output_shape is None:
                raise ValueError("The HOGP surrogate needs the problem's output_shape.")
            return build_hogp(train_x, train_y, self.output_shape, step_limit=cfg.hogp_steps, lr=cfg.hogp_lr)
        raise ValueError(cfg.surrogate)

    def archive(self, model, train_x, train_y) -> torch.Tensor:
        """Outcomes that novelty is measured against."""
        if self.config.archive == "observed":
            return train_y
        return posterior_mean(model, train_x).to(train_y)

    # ------------------------------------------------------------------
    def suggest(self, train_x, train_y, *, bounds=None, candidates=None, available=None, seed=None, iteration=None):
        """Propose the next input.

        Continuous problems pass ``bounds`` (``2 x d``) and receive a ``1 x d``
        tensor. Discrete problems pass ``candidates`` (``N x d``) and a boolean
        ``available`` mask and receive the index of the chosen candidate.
        """
        model = self.fit(train_x, train_y, seed=seed, iteration=iteration)
        archive = self.archive(model, train_x, train_y)
        if candidates is not None:
            if available is None:
                available = torch.ones(candidates.shape[0], dtype=torch.bool, device=candidates.device)
            samples = self.sample_candidates(model, candidates)
            scores = candidate_scores(samples, archive, self.config.k)
            return self.select_candidate(scores, samples, available, train_y)
        if bounds is None:
            raise ValueError("Pass either bounds (continuous) or candidates (discrete).")
        return self.optimize_continuous(model, archive, bounds, train_y, seed=seed, iteration=iteration)

    # ------------------------------------------------------------------
    def sample_candidates(self, model, candidates) -> torch.Tensor:
        cfg = self.config
        method = resolve_discrete_method(cfg.discrete_sampler, candidates.shape[0], cfg.max_exact_candidates, model)
        return draw_discrete_samples(
            model, candidates, method=method, num_bases=cfg.num_bases, saas_sample_mode=cfg.saas_sample_mode
        )

    def select_candidate(self, scores, samples, available, train_y) -> int:
        masked = torch.where(available, scores, torch.full_like(scores, -torch.inf))
        if not bool(available.any()):
            raise RuntimeError("Every candidate has already been evaluated.")
        return int(torch.argmax(masked).item())

    def make_acquisition(self, model, archive) -> BeaconAcquisition:
        if self.config.surrogate == "hogp":
            raise NotImplementedError("The HOGP surrogate is only supported on discrete candidate sets.")
        return BeaconAcquisition(model, archive, k=self.config.k, num_bases=self.config.num_bases)

    def sample_pool(self, model, archive, bounds):
        """Thompson-sample outcomes on a scrambled Sobol pool in ``bounds``.

        Uses the pathwise sampler when the kernel supports it and exact joint
        sampling otherwise. Returns ``(pool, samples, scores)``.
        """
        sobol = SobolEngine(bounds.shape[-1], scramble=True, seed=int(torch.randint(0, 2**31 - 1, (1,)).item()))
        lower, upper = bounds
        pool = lower + (upper - lower) * sobol.draw(self.config.pool_size).to(bounds)
        with torch.no_grad():
            if pathwise_supported(model):
                samples = self.make_acquisition(model, archive).sample_outcomes(pool.unsqueeze(-2)).squeeze(-2)
            else:
                samples = draw_discrete_samples(model, pool, method="exact")
            scores = candidate_scores(samples, archive, self.config.k)
        return pool, samples, scores

    def optimize_continuous(self, model, archive, bounds, train_y, *, seed=None, iteration=None):
        cfg = self.config
        if not pathwise_supported(model):
            # No differentiable sample path (e.g. Tanimoto kernel): pick the
            # most novel point of a Sobol pool under an exact joint sample.
            pool, _, scores = self.sample_pool(model, archive, bounds)
            return pool[int(torch.argmax(scores))].unsqueeze(0)
        acq = self.make_acquisition(model, archive)
        candidate, _ = optimize_acqf_quiet(
            acq_function=acq,
            bounds=bounds,
            q=1,
            num_restarts=cfg.num_restarts,
            raw_samples=cfg.raw_samples,
            options={"maxiter": cfg.maxiter, "batch_limit": cfg.batch_limit},
            seed=seed,
            iteration=iteration,
            model=model,
        )
        return candidate.detach()


class BEACONSAAS(BEACON):
    """BEACON with SAAS fully Bayesian GPs for high-dimensional inputs.

    Each outcome gets a ``SaasFullyBayesianSingleTaskGP`` whose half-Cauchy
    lengthscale prior switches off irrelevant inputs. Hyperparameters are
    sampled with NUTS (``warmup_steps``, ``num_samples``, ``thinning``). A
    Thompson sample draws one MCMC hyperparameter sample uniformly at random
    and then a GP sample under it (``saas_sample_mode='random'``).
    """

    name = "beacon-saas"
    surrogate_override = "saas"


class UGBEACON(BEACON):
    """User-guided BEACON (paper Sec. 4.5).

    Given a partition ``G`` of outcome space, the feasible set is
    ``{x : g(x) not in any visited or excluded region}``; BEACON's acquisition
    is maximized over that set. If no candidate is feasible (all regions the
    sample can reach are visited), the unconstrained BEACON choice is used.

    Args:
        partition: Behaviour regions (e.g. the reachability grid, the
            non-uniform oil-sorbent boxes, or MNIST digit classes).
    """

    name = "ug-beacon"

    def __init__(self, config: Optional[BeaconConfig] = None, *, partition: OutcomePartition = None, **kwargs):
        if partition is None:
            raise ValueError("UG-BEACON needs an outcome partition (the problem's behaviour regions).")
        super().__init__(config, **kwargs)
        self.partition = partition
        self.last_fallback = False

    def blocked_regions(self, train_y) -> torch.Tensor:
        blocked = self.partition.visited(train_y).clone()
        for region in self.config.ug_excluded_regions:
            blocked[int(region)] = True
        return blocked

    def feasible(self, samples, train_y) -> torch.Tensor:
        blocked = self.blocked_regions(train_y)
        hits = self.partition.membership(samples).cpu() & blocked
        return (~hits.any(dim=-1)).to(samples.device)

    def select_candidate(self, scores, samples, available, train_y) -> int:
        feasible = available & self.feasible(samples, train_y)
        self.last_fallback = not bool(feasible.any())
        return super().select_candidate(scores, samples, available if self.last_fallback else feasible, train_y)

    def optimize_continuous(self, model, archive, bounds, train_y, *, seed=None, iteration=None):
        pool, samples, scores = self.sample_pool(model, archive, bounds)
        feasible = self.feasible(samples, train_y)
        self.last_fallback = not bool(feasible.any())
        if self.last_fallback:
            return super().optimize_continuous(model, archive, bounds, train_y, seed=seed, iteration=iteration)
        masked = torch.where(feasible, scores, torch.full_like(scores, -torch.inf))
        return pool[int(torch.argmax(masked))].unsqueeze(0)


ALGORITHMS = {cls.name: cls for cls in (BEACON, UGBEACON, BEACONSAAS)}

_ALIASES = {
    "beacon": "beacon",
    "ug-beacon": "ug-beacon",
    "ugbeacon": "ug-beacon",
    "user-guided-beacon": "ug-beacon",
    "beacon-saas": "beacon-saas",
    "beaconsaas": "beacon-saas",
    "saas": "beacon-saas",
    "saas-beacon": "beacon-saas",
}


def canonical_name(name: str) -> str:
    """Normalize an algorithm name (case-insensitive; ``_``/space ~ ``-``)."""
    key = name.strip().lower().replace("_", "-").replace(" ", "-")
    if key not in _ALIASES:
        raise ValueError(f"Unknown algorithm '{name}'. Choose from {sorted(ALGORITHMS)}.")
    return _ALIASES[key]


def get_algorithm(name: str, config: Optional[BeaconConfig] = None, **kwargs) -> BEACON:
    """Instantiate an algorithm by name: ``beacon``, ``ug-beacon`` or ``beacon-saas``."""
    return ALGORITHMS[canonical_name(name)](config, **kwargs)
