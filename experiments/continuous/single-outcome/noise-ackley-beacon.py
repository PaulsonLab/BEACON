#!/usr/bin/env python3
"""Noisy 4D Ackley: BEACON paper experiment.

Equivalent to ``python -m beacon run --problem noisy-ackley --algorithm beacon --noise-std 0.5``.
Paper settings: noise std in {0.5, 1, 2, 4}, 50 initial points, 300 evaluations, 50 bins.
Add `--set likelihood=noiseless archive=observed` for the paper's noise-unaware variant.

Any extra flags are forwarded and override the defaults, e.g.::

    python experiments/continuous/single-outcome/noise-ackley-beacon.py --device cuda --replicates 5 --iterations 50
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from beacon.cli import script_main  # noqa: E402

if __name__ == "__main__":
    sys.exit(script_main("--problem", "noisy-ackley", "--algorithm", "beacon", "--noise-std", "0.5"))
