# BEACON

Companion code for [BEACON: A Bayesian Optimization Inspired Strategy for Efficient Novelty Search](https://arxiv.org/abs/2406.03616).

BEACON uses Bayesian optimization ideas to search for novel behaviors in expensive black-box systems. This repository is organized as a paper companion: the scripts preserve the experiment structure used for the manuscript, while shared data, saved results, and plotting inputs live in predictable top-level folders.

## Repository Map

| Path | Contents |
| --- | --- |
| `Continuous_SingleOutcome/` | Ackley, Rosenbrock, Styblinski-Tang, and noisy Ackley BEACON experiment scripts. |
| `Continuous_MultiOutcome/` | Synthetic multi-outcome and maze experiment scripts. |
| `Discrete_SingleOutcome_Material/` | Material and molecule single-outcome experiment scripts. |
| `Discrete_MultiOutcome/` | Oil sorbent, joint gas uptake, and MNIST multi-outcome experiment scripts. |
| `Plotting/` | Figure-generation scripts that read saved tensors or `.mat` figure inputs. |
| `src/` | Shared utilities, including efficient Thompson sampling and repo-relative paths. |
| `data/materials/` | Tracked tabular and pickle inputs used by material-discovery scripts. |
| `results/continuous_single_outcome/` | Tracked saved tensor outputs for continuous single-outcome plots. |
| `results/figure_inputs/` | Tracked `.mat` files and figure assets used by plotting scripts. |
| `models/mnist/` | Tracked CNN/VAE weights used by the MNIST script. |
| `tests/` | Lightweight repository checks; these do not run full experiments. |

## Installation

Create an environment with Python 3.11 or a compatible version, then install the dependencies:

```sh
pip install -r requirements.txt
```

Some experiments are computationally heavy or require optional environments. In particular, the maze script uses `gymnasium-robotics`, and the MNIST script can download MNIST data into `data/mnist/`.

## Running Scripts

Run scripts from the repository root so their repo-relative paths resolve consistently:

```sh
python Continuous_SingleOutcome/Ackley_BEACON.py
python Discrete_SingleOutcome_Material/LogD_BEACON.py
python Plotting/Plotting_Synthetic_FINAL.py
```

The optimization scripts are intended to reproduce paper experiments and may take a long time. Most `torch.save(...)` calls are left commented so the checked-in result artifacts are not overwritten accidentally. Uncomment or redirect saves only when regenerating results intentionally.

Plotting scripts read from `results/continuous_single_outcome/` or `results/figure_inputs/` and write figures into the current working directory.

## Shared Code

`src/ThompsonSampling.py` implements the efficient Thompson sampling strategy used by BEACON, based on Wilson et al., [Efficiently Sampling Functions from Gaussian Process Posteriors](https://arxiv.org/abs/2002.09309).

`src/paths.py` defines repo-root-relative paths used by scripts:

```python
from src.paths import MATERIALS_DIR, CONTINUOUS_SINGLE_OUTCOME_RESULTS_DIR
```

## Checks

The lightweight tests check syntax, artifact placement, absence of stale absolute paths, and removal of tracked bytecode:

```sh
python -B -m pytest
```

These checks deliberately do not run the full BEACON experiments.

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
