"""Benchmark problems from the BEACON paper.

Use :func:`get_problem` to build a problem by name, e.g.
``get_problem("ackley", dim=8)`` or ``get_problem("oil-sorbent")``, and
:func:`list_problems` to see what is available. New problems can be added
with :func:`register_problem` (see README, "Custom problems").
"""

from __future__ import annotations

from typing import Callable, Dict

from beacon.problems.base import ContinuousProblem, DiscreteProblem, Problem, minmax_normalize

_REGISTRY: Dict[str, Callable[..., Problem]] = {}
_DESCRIPTIONS: Dict[str, str] = {}


def register_problem(name: str, factory: Callable[..., Problem], description: str = "") -> None:
    """Register ``factory(**kwargs) -> Problem`` under ``name``."""
    _REGISTRY[name] = factory
    _DESCRIPTIONS[name] = description


def get_problem(name: str, **kwargs) -> Problem:
    """Instantiate a registered problem (keyword arguments go to its factory)."""
    key = name.strip().lower().replace("_", "-")
    if key not in _REGISTRY:
        raise ValueError(f"Unknown problem '{name}'. Available: {', '.join(sorted(_REGISTRY))}")
    return _REGISTRY[key](**kwargs)


def list_problems() -> Dict[str, str]:
    """``{name: description}`` of all registered problems."""
    return dict(sorted(_DESCRIPTIONS.items()))


def _lazy(module: str, attr: str, **fixed):
    def factory(**kwargs):
        import importlib

        return getattr(importlib.import_module(module), attr)(**fixed, **kwargs)

    return factory


_BUILTINS = [
    ("ackley", "beacon.problems.synthetic", "make_synthetic", {"name": "ackley"},
     "Ackley on [-5,5]^d (kwargs: dim=4|8|12, n_bins=25). Continuous, 1 outcome."),
    ("rosenbrock", "beacon.problems.synthetic", "make_synthetic", {"name": "rosenbrock"},
     "Rosenbrock on [-5,5]^d (kwargs: dim, n_bins). Continuous, 1 outcome."),
    ("styblinski-tang", "beacon.problems.synthetic", "make_synthetic", {"name": "styblinski-tang"},
     "Styblinski-Tang on [-5,5]^d (kwargs: dim, n_bins). Continuous, 1 outcome."),
    ("noisy-ackley", "beacon.problems.synthetic", "make_noisy_ackley", {},
     "4D Ackley with observation noise (kwargs: noise_std=0.5, n_bins=50)."),
    ("multi-output-plus", "beacon.problems.synthetic", "make_multi_output_plus", {},
     "6D synthetic problem with 2 outcomes, 10x10 grid."),
    ("maze", "beacon.problems.maze", "make_maze", {},
     "Point-maze RL policy search (8D, 2 outcomes; needs gymnasium-robotics)."),
    ("hydrogen", "beacon.problems.datasets", "make_hydrogen", {},
     "H2 uptake of 98,694 MOFs (7 descriptors). Discrete."),
    ("nitrogen", "beacon.problems.datasets", "make_nitrogen", {},
     "N2 uptake of 5,224 MOFs (20 descriptors). Discrete."),
    ("joint-gas", "beacon.problems.datasets", "make_joint_gas", {},
     "Joint CO2/CH4 uptake of 7,000 MOFs (25 descriptors, 2 outcomes). Discrete."),
    ("water-solubility", "beacon.problems.datasets", "make_water_solubility", {},
     "Water solubility of 900 molecules (14 descriptors). Discrete."),
    ("esol", "beacon.problems.datasets", "make_esol", {},
     "ESOL solubility, fingerprints + Tanimoto GP (needs gauche/rdkit). Discrete."),
    ("logd", "beacon.problems.datasets", "make_logd", {},
     "LogD of ~2k molecules, 125 descriptors (BEACON-SAAS study). Discrete."),
    ("oil-sorbent", "beacon.problems.oil_sorbent", "make_oil_sorbent", {},
     "Oil sorbent, 7 inputs -> 2 outcomes, non-uniform partition (UG-BEACON study). Discrete."),
    ("mnist", "beacon.problems.mnist", "make_mnist", {},
     "MNIST digits from a VAE latent space, HOGP surrogate (UG-BEACON study). Discrete."),
]

for _name, _module, _attr, _fixed, _desc in _BUILTINS:
    register_problem(_name, _lazy(_module, _attr, **_fixed), _desc)

__all__ = [
    "ContinuousProblem",
    "DiscreteProblem",
    "Problem",
    "get_problem",
    "list_problems",
    "minmax_normalize",
    "register_problem",
]
