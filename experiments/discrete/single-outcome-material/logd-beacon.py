#!/usr/bin/env python3
"""LogD with BEACON (discrete): BEACON paper experiment.

Equivalent to ``python -m beacon run --problem logd --algorithm beacon``.
Paper settings: 125 descriptors, 10 initial points, 100 evaluations, 20 replicates.

Any extra flags are forwarded and override the defaults, e.g.::

    python experiments/discrete/single-outcome-material/logd-beacon.py --device cuda --replicates 5 --iterations 50
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from beacon.cli import script_main  # noqa: E402

if __name__ == "__main__":
    sys.exit(script_main("--problem", "logd", "--algorithm", "beacon"))
