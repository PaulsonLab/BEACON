"""BEACON: Bayesian-optimization-inspired novelty search.

Quick start::

    from beacon import run_experiment
    result = run_experiment("ackley", "beacon", problem_kwargs={"dim": 4},
                            overrides={"iterations": 50, "replicates": 2}, device="auto")
    print(result.summary())

or, step by step on your own data::

    from beacon import get_algorithm
    algo = get_algorithm("beacon", k=10)
    x_next = algo.suggest(train_x, train_y, bounds=bounds)          # continuous
    idx = algo.suggest(train_x, train_y, candidates=X, available=m)  # discrete
"""

__version__ = "1.0.0"

from beacon.algorithms import ALGORITHMS, BEACON, BEACONSAAS, UGBEACON, BeaconConfig, get_algorithm
from beacon.device import resolve_device, set_seed
from beacon.problems import ContinuousProblem, DiscreteProblem, get_problem, list_problems, register_problem
from beacon.runner import run_experiment

__all__ = [
    "ALGORITHMS",
    "BEACON",
    "BEACONSAAS",
    "UGBEACON",
    "BeaconConfig",
    "ContinuousProblem",
    "DiscreteProblem",
    "get_algorithm",
    "get_problem",
    "list_problems",
    "register_problem",
    "resolve_device",
    "run_experiment",
    "set_seed",
    "__version__",
]
