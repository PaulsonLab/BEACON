#!/usr/bin/env python3
"""MNIST digits with UG-BEACON (digit classes as regions): BEACON paper experiment.

Equivalent to ``python -m beacon run --problem mnist --algorithm ug-beacon``.
Paper settings: 10 initial points, 40 evaluations, 10 replicates. A GPU is recommended.

Any extra flags are forwarded and override the defaults, e.g.::

    python experiments/discrete/multi-outcome/mnist/mnist-ug-beacon.py --device cuda --replicates 5 --iterations 50
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from beacon.cli import script_main  # noqa: E402

if __name__ == "__main__":
    sys.exit(script_main("--problem", "mnist", "--algorithm", "ug-beacon"))
