"""Run BEACON experiments (replicates of the novelty-search loop).

Outputs (``output_dir``):

* ``cost.pt``       - ``(replicates, iterations + 1)`` evaluations spent
  (excluding the initial design), as in the original scripts.
* ``coverage.pt``   - same shape, reachability after each evaluation
  (``best_reward.pt`` for the maze problem).
* ``time.pt``       - wall-clock seconds per iteration.
* ``history.pt``    - per replicate: selected inputs, observed and noise-free
  outcomes (and candidate indices for discrete problems).
* ``config.json``   - problem, algorithm, all settings, device and package
  versions, for provenance.
"""

from __future__ import annotations

import json
import platform
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import torch

from beacon.algorithms import BEACON, BeaconConfig, canonical_name, get_algorithm
from beacon.device import DTYPE, resolve_device, set_seed
from beacon.paths import GENERATED_RESULTS_DIR
from beacon.problems import Problem, get_problem

EXPERIMENT_KEYS = ("n_init", "iterations", "replicates", "seed", "early_stop")


@dataclass
class ExperimentSettings:
    """Loop-level settings (BEACON's own options live in :class:`BeaconConfig`)."""

    n_init: int = 10
    iterations: int = 100
    replicates: int = 1
    seed: int = 0
    """Replicate ``r`` uses seed ``seed + r`` (paper: seeds 0..replicates-1)."""
    early_stop: bool = True
    """Stop evaluating once reachability reaches 1 (the curve stays at 1)."""


@dataclass
class ExperimentResult:
    problem: str
    algorithm: str
    metric_name: str
    cost: torch.Tensor
    metric: torch.Tensor
    times: torch.Tensor
    histories: List[Dict] = field(default_factory=list)
    settings: Dict = field(default_factory=dict)
    output_dir: Optional[Path] = None

    @property
    def final_metric(self) -> torch.Tensor:
        return self.metric[:, -1]

    def summary(self) -> str:
        final = self.final_metric
        std = final.std().item() if final.numel() > 1 else 0.0
        return (
            f"{self.algorithm} on {self.problem}: final {self.metric_name} = "
            f"{final.mean().item():.3f} +/- {std:.3f} over {final.numel()} replicate(s)"
        )


def resolve_settings(problem: Problem, overrides: Optional[Dict] = None):
    """Merge problem defaults with user overrides; split loop vs BEACON options."""
    merged = {**problem.defaults, **{k: v for k, v in (overrides or {}).items() if v is not None}}
    unknown = set(merged) - set(EXPERIMENT_KEYS) - BeaconConfig.field_names()
    if unknown:
        raise ValueError(f"Unknown setting(s): {sorted(unknown)}")
    settings = ExperimentSettings(**{k: merged[k] for k in EXPERIMENT_KEYS if k in merged})
    config = BeaconConfig().replace(**{k: v for k, v in merged.items() if k in BeaconConfig.field_names()})
    return settings, config


def make_algorithm(name: str, problem: Problem, config: BeaconConfig) -> BEACON:
    name = canonical_name(name)
    kwargs = {"output_shape": problem.output_shape}
    if name == "ug-beacon":
        if problem.guidance_partition is None:
            raise ValueError(f"Problem '{problem.name}' defines no behaviour partition for UG-BEACON.")
        kwargs["partition"] = problem.guidance_partition
    return get_algorithm(name, config, **kwargs)


def _is_complete(problem: Problem, value: float) -> bool:
    return problem.metric_name == "reachability" and value >= 1.0 - 1e-12


def run_replicate(problem: Problem, algorithm: BEACON, settings: ExperimentSettings, seed: int, device, verbose: int = 1):
    """One replicate of the BEACON loop. Returns ``(metric_list, times, history)``."""
    set_seed(seed)
    history: Dict = {"seed": seed}
    times: List[float] = []
    fallbacks = 0

    if problem.is_discrete:
        candidates = problem.X.to(device=device, dtype=DTYPE)
        targets = problem.Y.to(device=device, dtype=DTYPE)
        ids = list(int(i) for i in problem.initial_indices(settings.n_init, seed))
        available = torch.ones(problem.n_candidates, dtype=torch.bool, device=device)
        available[ids] = False
        metric = [problem.metric(problem.Y_true[ids])]
    else:
        X, Y, Y_true = problem.initial_data(settings.n_init, seed, device=device)
        bounds = problem.unit_bounds(device)
        metric = [problem.metric(Y_true)]

    for it in range(settings.iterations):
        if settings.early_stop and _is_complete(problem, metric[-1]):
            metric.append(metric[-1])
            times.append(0.0)
            continue
        start = time.perf_counter()
        if problem.is_discrete:
            idx = algorithm.suggest(
                candidates[ids], targets[ids], candidates=candidates, available=available, seed=seed, iteration=it
            )
            ids.append(idx)
            available[idx] = False
            metric.append(problem.metric(problem.Y_true[ids]))
        else:
            x_new = algorithm.suggest(X, Y, bounds=bounds, seed=seed, iteration=it).to(X)
            y_obs, y_true = problem.evaluate(x_new)
            X, Y, Y_true = torch.cat([X, x_new]), torch.cat([Y, y_obs]), torch.cat([Y_true, y_true])
            metric.append(problem.metric(Y_true))
        times.append(time.perf_counter() - start)
        fallbacks += int(getattr(algorithm, "last_fallback", False))
        if verbose >= 2:
            print(f"  seed={seed} iter={it + 1}/{settings.iterations} {problem.metric_name}={metric[-1]:.4f} ({times[-1]:.2f}s)")

    if problem.is_discrete:
        history.update(indices=torch.tensor(ids), X=problem.X[ids], Y=problem.Y[ids], Y_true=problem.Y_true[ids])
    else:
        history.update(X=X.cpu(), Y=Y.cpu(), Y_true=Y_true.cpu())
    if fallbacks:
        history["ug_fallbacks"] = fallbacks
    return metric, times, history


def run_experiment(
    problem: Union[str, Problem],
    algorithm: str = "beacon",
    *,
    problem_kwargs: Optional[Dict] = None,
    overrides: Optional[Dict] = None,
    device="auto",
    output_dir: Union[str, Path, None] = "default",
    verbose: int = 1,
) -> ExperimentResult:
    """Run ``replicates`` independent BEACON runs and (optionally) save them.

    Args:
        problem: Problem instance or registered name (see ``beacon list``).
        algorithm: ``beacon``, ``ug-beacon`` or ``beacon-saas``.
        problem_kwargs: Passed to the problem factory (e.g. ``{"dim": 8}``).
        overrides: Experiment settings (``n_init``, ``iterations``,
            ``replicates``, ``seed``, ``early_stop``) and/or
            :class:`BeaconConfig` fields; ``None`` values are ignored.
        device: ``"auto"``, ``"cpu"``, ``"cuda"``, ``"cuda:1"``, ...
        output_dir: Directory for results; ``"default"`` uses
            ``results/generated/<problem>/<algorithm>``; ``None`` saves nothing.
        verbose: 0 silent, 1 per replicate, 2 per iteration.
    """
    if isinstance(problem, str):
        problem = get_problem(problem, **(problem_kwargs or {}))
    algorithm = canonical_name(algorithm)
    device = resolve_device(device)
    settings, config = resolve_settings(problem, overrides)
    algo = make_algorithm(algorithm, problem, config)

    if verbose:
        print(f"Running {algorithm} on {problem.name} ({settings.replicates} replicate(s) x "
              f"{settings.iterations} iterations, n_init={settings.n_init}) on {device}")

    metrics, all_times, histories = [], [], []
    for r in range(settings.replicates):
        seed = settings.seed + r
        metric, times, history = run_replicate(problem, algo, settings, seed, device, verbose)
        metrics.append(metric)
        all_times.append(times)
        histories.append(history)
        if verbose:
            print(f"  replicate {r + 1}/{settings.replicates} (seed {seed}): {problem.metric_name} "
                  f"{metric[0]:.3f} -> {metric[-1]:.3f} in {sum(times):.1f}s")

    n_steps = settings.iterations + 1
    result = ExperimentResult(
        problem=problem.name,
        algorithm=algorithm,
        metric_name=problem.metric_name,
        cost=torch.arange(n_steps, dtype=torch.float32).repeat(settings.replicates, 1),
        metric=torch.tensor(metrics, dtype=torch.float32),
        times=torch.tensor(all_times, dtype=torch.float32),
        histories=histories,
        settings={
            "problem": problem.name,
            "problem_kwargs": problem_kwargs or {},
            "algorithm": algorithm,
            "experiment": asdict(settings),
            "beacon": config.to_dict(),
            "device": str(device),
        },
    )
    if verbose:
        print(result.summary())
    if output_dir is not None:
        if output_dir == "default":
            output_dir = GENERATED_RESULTS_DIR / problem.name / algorithm
        save_result(result, Path(output_dir))
        if verbose:
            print(f"Saved results to {result.output_dir}")
    return result


def save_result(result: ExperimentResult, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    metric_file = "coverage.pt" if result.metric_name == "reachability" else f"{result.metric_name}.pt"
    torch.save(result.cost, output_dir / "cost.pt")
    torch.save(result.metric, output_dir / metric_file)
    torch.save(result.times, output_dir / "time.pt")
    torch.save(result.histories, output_dir / "history.pt")
    provenance = dict(result.settings)
    provenance["versions"] = _versions()
    provenance["final_metric_mean"] = float(result.final_metric.mean())
    (output_dir / "config.json").write_text(json.dumps(provenance, indent=2, default=str), encoding="utf-8")
    result.output_dir = output_dir


def _versions() -> Dict[str, str]:
    import botorch
    import gpytorch

    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "botorch": botorch.__version__,
        "gpytorch": gpytorch.__version__,
        "numpy": np.__version__,
    }
