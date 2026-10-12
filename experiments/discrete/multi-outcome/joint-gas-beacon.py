#!/usr/bin/env python3
"""Joint CO2/CH4 uptake of MOFs (discrete, 2 outcomes): BEACON paper experiment.

Equivalent to ``python -m beacon run --problem joint-gas --algorithm beacon``.
Paper settings: 50 initial points, 500 evaluations, 20 replicates, 10 x 10 grid.

Any extra flags are forwarded and override the defaults, e.g.::

    python experiments/discrete/multi-outcome/joint-gas-beacon.py --device cuda --replicates 5 --iterations 50
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from beacon.cli import script_main  # noqa: E402

if __name__ == "__main__":
    sys.exit(script_main("--problem", "joint-gas", "--algorithm", "beacon"))
