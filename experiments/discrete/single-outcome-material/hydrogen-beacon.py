#!/usr/bin/env python3
"""Hydrogen uptake of MOFs (discrete): BEACON paper experiment.

Equivalent to ``python -m beacon run --problem hydrogen --algorithm beacon``.
Paper settings: 10 initial points, 80 evaluations, 20 replicates.

Any extra flags are forwarded and override the defaults, e.g.::

    python experiments/discrete/single-outcome-material/hydrogen-beacon.py --device cuda --replicates 5 --iterations 50
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from beacon.cli import script_main  # noqa: E402

if __name__ == "__main__":
    sys.exit(script_main("--problem", "hydrogen", "--algorithm", "beacon"))
