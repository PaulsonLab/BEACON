#!/usr/bin/env python3
"""Styblinski-Tang (continuous, single outcome): BEACON paper experiment.

Equivalent to ``python -m beacon run --problem styblinski-tang --algorithm beacon --dim 12``.
Paper settings: d in {4, 8, 12}, 10 initial points, 200 evaluations, 20 replicates, 25 bins.

Any extra flags are forwarded and override the defaults, e.g.::

    python experiments/continuous/single-outcome/styblinski-tang-beacon.py --device cuda --replicates 5 --iterations 50
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from beacon.cli import script_main  # noqa: E402

if __name__ == "__main__":
    sys.exit(script_main("--problem", "styblinski-tang", "--algorithm", "beacon", "--dim", "12"))
