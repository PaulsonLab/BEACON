#!/usr/bin/env python3
"""Oil sorbent with BEACON (discrete, 2 outcomes): BEACON paper experiment.

Equivalent to ``python -m beacon run --problem oil-sorbent --algorithm beacon``.
Paper settings: 10 initial points, 200 evaluations, 10 replicates, k = 3.

Any extra flags are forwarded and override the defaults, e.g.::

    python experiments/discrete/multi-outcome/oil-beacon.py --device cuda --replicates 5 --iterations 50
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from beacon.cli import script_main  # noqa: E402

if __name__ == "__main__":
    sys.exit(script_main("--problem", "oil-sorbent", "--algorithm", "beacon"))
