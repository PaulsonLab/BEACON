#!/usr/bin/env python3
"""Discrete example: discovering diverse nitrogen uptakes in a MOF database.

A database lists 5,224 metal-organic frameworks (MOFs) with 20 structural and
chemical descriptors and their N2 uptake (Daglar et al.; paper Sec. 5.2).
Each "evaluation" reveals one MOF's uptake (in practice: a simulation or a
measurement). BEACON chooses which untested MOF to evaluate next, aiming to
cover as many distinct uptake levels as possible.

This mirrors how you would use BEACON on your own library: load a table, build
descriptor (X) and property (Y) tensors, and either

* Part A - drive the campaign yourself with ``suggest(..., candidates=...)``, or
* Part B - wrap the table as a ``DiscreteProblem`` for ``run_experiment``.

Run:  python examples/discrete_nitrogen_mofs.py [--device cuda] [--quick]
"""

import argparse

import numpy as np
import pandas as pd
import torch

from beacon import DiscreteProblem, get_algorithm, run_experiment
from beacon.metrics import HistogramPartition, Reachability
from beacon.paths import MATERIALS_DIR


def load_mofs():
    """Return MOF names, min-max scaled descriptors X (N, 20) and N2 uptake Y (N, 1)."""
    df = pd.read_csv(MATERIALS_DIR / "Nitrogen.csv", encoding="utf-8-sig")
    descriptors = df.columns[1:21]  # LCD, PLD, density, pore volume, ..., elemental composition
    X = torch.tensor(df[descriptors].to_numpy(dtype=np.float64))
    X = (X - X.min(dim=0).values) / (X.max(dim=0).values - X.min(dim=0).values)
    Y = torch.tensor(df["U_N2 (mol/kg)"].to_numpy(dtype=np.float64)).unsqueeze(-1)
    return df["MOF"].tolist(), X, Y


# ---------------------------------------------------------------------------
# Part A: ask-tell over a candidate library
# ---------------------------------------------------------------------------
def ask_tell(n_init: int = 10, iterations: int = 30, device: str = "cpu"):
    torch.manual_seed(0)
    names, X, Y = load_mofs()
    X, Y = X.to(device), Y.to(device)

    rng = np.random.RandomState(0)
    tested = [int(i) for i in rng.choice(len(X), size=n_init, replace=False)]  # initial screen
    available = torch.ones(len(X), dtype=torch.bool, device=device)
    available[tested] = False

    # Matern-5/2 GP, as in the paper's MOF studies.
    algo = get_algorithm("beacon", k=10, kernel="matern", archive="observed")
    for it in range(iterations):
        idx = algo.suggest(X[tested], Y[tested], candidates=X, available=available)
        # In a real campaign: simulate or measure MOF `names[idx]` here, then record Y[idx].
        tested.append(idx)
        available[idx] = False
        print(f"  eval {it + 1:3d}: {names[idx]:<28s} N2 uptake = {Y[idx, 0].item():.3f} mol/kg")
    return [names[i] for i in tested], Y[tested].cpu()


# ---------------------------------------------------------------------------
# Part B: a reusable problem definition for the experiment runner
# ---------------------------------------------------------------------------
def make_problem(iterations: int = 100, replicates: int = 3) -> DiscreteProblem:
    _, X, Y = load_mofs()
    # Behaviour regions: 25 equal-width uptake bins. Reachability is normalized by the
    # bins the whole database occupies, so 1.0 means every attainable uptake level was found.
    partition = HistogramPartition(float(Y.min()), float(Y.max()), n_bins=25)
    return DiscreteProblem(
        "nitrogen-mofs",
        X,
        Y,
        metric=Reachability.from_dataset(partition, Y),
        partition=partition,  # also the behaviour regions for UG-BEACON
        defaults=dict(n_init=10, iterations=iterations, replicates=replicates, k=10, kernel="matern", archive="observed"),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--device", default="auto")
    parser.add_argument("--quick", action="store_true", help="tiny budgets (smoke test)")
    args = parser.parse_args()
    budget = dict(iterations=3, replicates=1) if args.quick else dict(iterations=100, replicates=3)

    print("Part A: ask-tell loop")
    from beacon.device import resolve_device

    tested, uptake = ask_tell(iterations=3 if args.quick else 30, device=resolve_device(args.device))
    print(f"  tested {len(tested)} MOFs; uptake range [{uptake.min():.3f}, {uptake.max():.3f}] mol/kg")

    print("\nPart B: BEACON vs UG-BEACON with the experiment runner")
    problem = make_problem(**budget)
    for algorithm in ("beacon", "ug-beacon"):
        result = run_experiment(problem, algorithm, device=args.device, output_dir=None, verbose=0)
        print(" ", result.summary())


if __name__ == "__main__":
    main()
