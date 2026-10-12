#!/usr/bin/env python3
"""LogD with BEACON-SAAS (discrete): BEACON paper experiment.

Equivalent to ``python -m beacon run --problem logd --algorithm beacon-saas --set saas_sample_mode=mean``.
Paper settings: NUTS with 128 warm-up steps, 64 samples, thinning 16.
`saas_sample_mode=mean` reproduces the original script; drop it for a single-MCMC-draw Thompson sample.

Any extra flags are forwarded and override the defaults, e.g.::

    python experiments/discrete/single-outcome-material/logd-beacon-saas.py --device cuda --replicates 5 --iterations 50
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from beacon.cli import script_main  # noqa: E402

if __name__ == "__main__":
    sys.exit(script_main("--problem", "logd", "--algorithm", "beacon-saas", "--set", "saas_sample_mode=mean"))
