"""Experiment groups that reproduce the BEACON paper's studies.

Each group is a list of runs ``(problem, problem_kwargs, algorithm,
overrides, tag)``; the problem defaults already encode the paper settings.
Run a group with ``python -m beacon reproduce <group>``.
"""

from __future__ import annotations

from typing import Dict, List, NamedTuple


class PaperRun(NamedTuple):
    problem: str
    problem_kwargs: dict
    algorithm: str
    overrides: dict
    tag: str


def _synthetic() -> List[PaperRun]:
    return [
        PaperRun(name, {"dim": dim}, "beacon", {}, f"{name}-{dim}d")
        for dim in (4, 8, 12)
        for name in ("ackley", "rosenbrock", "styblinski-tang")
    ]


def _noise() -> List[PaperRun]:
    runs = []
    for noise in (0.5, 1.0, 2.0, 4.0):
        kw = {"noise_std": noise}
        runs.append(PaperRun("noisy-ackley", kw, "beacon", {}, f"noisy-ackley-{noise}-considernoise"))
        runs.append(
            PaperRun(
                "noisy-ackley", kw, "beacon", {"likelihood": "noiseless", "archive": "observed"},
                f"noisy-ackley-{noise}-considernoiseless",
            )
        )
    return runs


PAPER_EXPERIMENTS: Dict[str, List[PaperRun]] = {
    "synthetic": _synthetic(),
    "noise": _noise(),
    "multi-output-plus": [PaperRun("multi-output-plus", {}, "beacon", {}, "multi-output-plus")],
    "maze": [PaperRun("maze", {}, "beacon", {}, "maze")],
    "materials": [
        PaperRun("hydrogen", {}, "beacon", {}, "hydrogen"),
        PaperRun("nitrogen", {}, "beacon", {}, "nitrogen"),
        PaperRun("joint-gas", {}, "beacon", {}, "joint-gas"),
    ],
    "molecules": [
        PaperRun("water-solubility", {}, "beacon", {}, "water-solubility"),
        PaperRun("esol", {}, "beacon", {}, "esol"),
        PaperRun("logd", {}, "beacon", {}, "logd"),
        # The original script averaged samples over MCMC draws; keep that here.
        PaperRun("logd", {}, "beacon-saas", {"saas_sample_mode": "mean"}, "logd-saas"),
    ],
    "oil-sorbent": [
        PaperRun("oil-sorbent", {}, "beacon", {}, "oil-sorbent"),
        PaperRun("oil-sorbent", {}, "ug-beacon", {}, "oil-sorbent-ug"),
    ],
    "mnist": [
        PaperRun("mnist", {}, "beacon", {}, "mnist"),
        PaperRun("mnist", {}, "ug-beacon", {}, "mnist-ug"),
    ],
}

#: Names of the experiment groups (as opposed to single-run tags).
PAPER_GROUPS = tuple(PAPER_EXPERIMENTS)

# Individual runs are addressable by their tag as well (e.g. "ackley-4d").
for _group in list(PAPER_EXPERIMENTS.values()):
    for _run in _group:
        PAPER_EXPERIMENTS.setdefault(_run.tag, [_run])
