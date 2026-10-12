#!/usr/bin/env python3
"""Continuous example: exploring achievable oil-sorbent properties with BEACON.

Seven process variables (polymer ratios, concentrations, temperatures, ...) map
to two properties of an oil sorbent: oil-absorption capacity and mechanical
strength. The paper's case study (Sec. 5.5) uses fitted response surfaces for
this map. Here the variables are treated as *continuous* within their physical
bounds, so BEACON optimizes its acquisition with gradients instead of
choosing from a fixed candidate list.

The script shows two ways to use BEACON:

* Part A - an ask-tell loop around your own evaluation function (a simulator
  or a lab experiment), using only ``get_algorithm(...).suggest``.
* Part B - wrapping the black box as a ``ContinuousProblem`` to get
  replicates, seeding, metrics and result files from ``run_experiment``.

Run:  python examples/continuous_oil_sorbent.py [--device cuda] [--quick]
"""

import argparse

import torch

from beacon import ContinuousProblem, get_algorithm, run_experiment
from beacon.metrics import GridPartition, Reachability
from beacon.problems.oil_sorbent import OIL_LOWER, OIL_UPPER, oil_sorbent

LOWER, UPPER = OIL_LOWER.double(), OIL_UPPER.double()  # physical bounds of the 7 variables
# Typical magnitudes of [capacity, strength]. BEACON measures novelty with Euclidean
# distances in outcome space, so outcomes should be on comparable scales.
SCALE = torch.tensor([2.0e5, 1.25e4], dtype=torch.double)


def sorbent_properties(X: torch.Tensor) -> torch.Tensor:
    """Black box: physical process variables (n, 7) -> scaled [capacity, strength] (n, 2)."""
    Y = oil_sorbent(X).clamp_min(0)  # negative predictions of the response surfaces are unphysical
    return Y / SCALE.to(Y)


# ---------------------------------------------------------------------------
# Part A: ask-tell loop (BEACON proposes, you evaluate)
# ---------------------------------------------------------------------------
def ask_tell(n_init: int = 10, iterations: int = 30, device: str = "cpu"):
    torch.manual_seed(0)
    tkw = {"dtype": torch.double, "device": device}
    lower, upper = LOWER.to(**tkw), UPPER.to(**tkw)

    def evaluate(x_unit):  # replace with your simulator or experiment
        return sorbent_properties(lower + (upper - lower) * x_unit)

    X = torch.rand(n_init, 7, **tkw)  # BEACON works on the unit cube [0, 1]^7
    Y = evaluate(X)
    bounds = torch.stack([torch.zeros(7, **tkw), torch.ones(7, **tkw)])
    algo = get_algorithm("beacon", k=10, kernel="rbf", archive="observed")

    for it in range(iterations):
        x_next = algo.suggest(X, Y, bounds=bounds)  # (1, 7) in [0, 1]^7
        y_next = evaluate(x_next)
        X, Y = torch.cat([X, x_next]), torch.cat([Y, y_next])
        cap, strength = (y_next[0] * SCALE.to(**tkw)).tolist()
        print(f"  eval {it + 1:3d}: capacity={cap:9.0f}  strength={strength:7.0f}")

    recipes = lower + (upper - lower) * X  # back to physical units
    return recipes, Y * SCALE.to(**tkw)


# ---------------------------------------------------------------------------
# Part B: a reusable problem definition for the experiment runner
# ---------------------------------------------------------------------------
def make_problem(iterations: int = 100, replicates: int = 3) -> ContinuousProblem:
    # Behaviour regions: a 10 x 10 grid over the scaled outcome plane.
    partition = GridPartition(lower=[0.0, 0.0], upper=[1.0, 1.0], n_bins=10)
    # Not every cell can be reached. Normalize reachability by the cells hit by a dense
    # random sample of the design space, so that 1.0 means "every reachable cell found".
    g = torch.Generator().manual_seed(0)
    X_dense = torch.rand(100_000, 7, generator=g, dtype=torch.double)
    reachable = sorbent_properties(LOWER + (UPPER - LOWER) * X_dense)
    return ContinuousProblem(
        name="oil-sorbent-continuous",
        objective=sorbent_properties,
        input_dim=7,
        output_dim=2,
        lower=LOWER,
        upper=UPPER,
        metric=Reachability.from_dataset(partition, reachable),
        partition=partition,  # also the behaviour regions for UG-BEACON
        defaults=dict(n_init=10, iterations=iterations, replicates=replicates, k=10, kernel="rbf", archive="observed"),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--device", default="auto")
    parser.add_argument("--quick", action="store_true", help="tiny budgets (smoke test)")
    args = parser.parse_args()
    budget = dict(iterations=3, replicates=1) if args.quick else dict(iterations=100, replicates=3)
    fast = dict(num_restarts=2, raw_samples=16, pool_size=256) if args.quick else {}

    print("Part A: ask-tell loop")
    from beacon.device import resolve_device

    recipes, props = ask_tell(iterations=3 if args.quick else 30, device=resolve_device(args.device))
    print(f"  evaluated {len(recipes)} recipes; capacity range "
          f"[{props[:, 0].min():.0f}, {props[:, 0].max():.0f}]")

    print("\nPart B: BEACON vs UG-BEACON with the experiment runner")
    problem = make_problem(**budget)
    for algorithm in ("beacon", "ug-beacon"):
        result = run_experiment(problem, algorithm, overrides=fast, device=args.device, output_dir=None, verbose=0)
        print(" ", result.summary())


if __name__ == "__main__":
    main()
