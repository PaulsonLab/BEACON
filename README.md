# BEACON

Companion code for [BEACON: A Bayesian Optimization Inspired Strategy for Efficient Novelty Search](https://arxiv.org/abs/2406.03616).

BEACON uses Bayesian optimization ideas to search for novel behaviors in expensive black-box systems. This repository is organized as a paper companion: experiment scripts remain close to the manuscript studies, while shared inputs, saved outputs, trained weights, and plotting data live in predictable top-level folders.

## Repository Map

| Path | Contents |
| --- | --- |
| `beacon/` | Shared Python helpers, including repo-relative paths and efficient Thompson sampling. |
| `experiments/continuous/single-outcome/` | Ackley, Rosenbrock, Styblinski-Tang, and noisy Ackley scripts. |
| `experiments/continuous/multi-outcome/` | Synthetic multi-outcome and maze scripts. |
| `experiments/discrete/single-outcome-material/` | Material and molecule single-outcome scripts. |
| `experiments/discrete/multi-outcome/` | Oil sorbent, joint gas uptake, and MNIST multi-outcome scripts. |
| `figures/scripts/` | Plotting scripts used to recreate paper-style figures from saved data. |
| `data/materials/` | Tracked tabular and pickle inputs used by material-discovery scripts. |
| `results/continuous-single-outcome/` | Tracked saved tensor outputs for continuous single-outcome plots. |
| `results/plot-data/` | Saved `.mat` files and image assets that plotting scripts read. |
| `models/mnist/` | Tracked CNN/VAE weights used by the MNIST script. |
| `tests/` | Lightweight repository checks; these do not run full experiments. |

`results/plot-data/` means "data used by plotting scripts," not generated figures. Generated figures should be treated as local outputs, and `figures/output/` is ignored by git for that purpose.

## Installation

Use a repo-local virtual environment:

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

On machines where the default `python3` is newer than the scientific stack supports, Python 3.11 is a good target. For example, with `uv` installed:

```sh
uv venv --python 3.11 --seed --clear .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Some experiments are computationally heavy or require optional environments. In particular, the maze script uses `gymnasium-robotics`, and the MNIST script can download MNIST data into `data/mnist/`.

## Running Scripts

Run scripts from the repository root so their repo-relative paths resolve consistently:

```sh
python experiments/continuous/single-outcome/ackley-beacon.py
python experiments/discrete/single-outcome-material/logd-beacon.py
python figures/scripts/plot-synthetic.py
```

The optimization scripts are intended to reproduce paper experiments and may take a long time. Most `torch.save(...)` calls are left commented so the checked-in result artifacts are not overwritten accidentally. Uncomment or redirect saves only when regenerating results intentionally.

Plotting scripts read from `results/continuous-single-outcome/` or `results/plot-data/` and write generated figures into `figures/output/`.

## Shared Code

`beacon/thompson_sampling.py` implements the efficient Thompson sampling strategy used by BEACON, based on Wilson et al., [Efficiently Sampling Functions from Gaussian Process Posteriors](https://arxiv.org/abs/2002.09309).

`beacon/paths.py` defines repo-root-relative paths used by scripts:

```python
from beacon.paths import MATERIALS_DIR, PLOT_DATA_DIR, FIGURE_OUTPUTS_DIR
```

## Checks

The lightweight tests check syntax, importability of shared helpers, artifact placement, stale absolute paths, naming conventions, and removal of tracked bytecode:

```sh
source .venv/bin/activate
python -B -m pytest
```

These checks deliberately do not run the full BEACON experiments, MNIST downloads, or maze simulations.

Before merging a cleanup branch back to `main`, confirm:

- `python -B -m pytest` passes in `.venv`.
- Artifact checksums match before and after structural moves.
- README commands match the actual paths.
- No active code imports retired `src.*` paths or references stale absolute paths.
- `git diff --name-status --find-renames repo-cleanup-companion...repo-structure-redesign` is readable.

## Extension To High-Dimensional Problems

The high-dimensional extension, TR-BEACON, is described in the [paper](https://openreview.net/pdf?id=9Xo6ONB8E3) and maintained in the [TR-BEACON repository](https://github.com/PaulsonLab/TR-BEACON).

## Citation

If you use this code in your research, please cite:

```bibtex
@article{tang2024beacon,
  title={Beacon: A bayesian optimization strategy for novelty search in expensive black-box systems},
  author={Tang, Wei-Ting and Chakrabarty, Ankush and Paulson, Joel A},
  journal={arXiv preprint arXiv:2406.03616},
  year={2024}
}
```
