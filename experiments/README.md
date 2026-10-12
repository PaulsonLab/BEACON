# BEACON paper experiments

This folder reproduces the studies in
[BEACON: A Bayesian Optimization Inspired Strategy for Efficient Novelty Search](https://doi.org/10.1039/d6dd00123h).
It describes the benchmark problems, the paper's experimental settings and evaluation metrics, and
how to regenerate the results and figures. To apply BEACON to your own problem, see the
[main README](../README.md).

**Contents**
[Running the experiments](#running-the-experiments) ·
[Benchmark problems](#benchmark-problems) ·
[Evaluation metrics](#evaluation-metrics) ·
[Reproducing the figures](#reproducing-the-figures)

Every command assumes the package is installed (`pip install -e .` from the repository root; see
[Installation](../README.md#installation)). Some studies need optional extras: `saas` for
BEACON-SAAS, `maze` for the maze and `chem` for ESOL.

---

## Running the experiments

The paper settings are the default settings of each problem, so the two ways of running an
experiment below are equivalent. Any setting can still be overridden with the usual flags
(`--device`, `--replicates`, `--iterations`, `--set KEY=VALUE`, …; see
[Configuration](../README.md#configuration)).

### Whole studies: `beacon reproduce`

`beacon reproduce <group>` runs a whole study with the paper settings:

| Group | Runs | Paper figure |
| --- | --- | --- |
| `synthetic` | Ackley / Rosenbrock / Styblinski-Tang × 4D, 8D, 12D | Synthetic benchmarks (`plot-synthetic.py`) |
| `noise` | Noisy Ackley, σ ∈ {0.5, 1, 2, 4}, noise-aware vs. noise-unaware | Noise study (`plot-noise.py`) |
| `multi-output-plus` | Multi-Output Plus | Multi-outcome synthetic (`plot-synthetic-multi-outcome.py`) |
| `maze` | Point maze | RL maze (`plot-maze.py`) |
| `materials` | Hydrogen, nitrogen, joint gas | Materials (`plot-mof.py`) |
| `molecules` | Water solubility, ESOL, LogD, LogD with BEACON-SAAS | Molecules (`plot-small-molecule.py`) |
| `oil-sorbent` | BEACON and UG-BEACON | Oil sorbent (`plot-oil-sorbent.py`) |
| `mnist` | BEACON and UG-BEACON | MNIST (`plot-mnist.py`) |

```sh
beacon reproduce synthetic --device cuda          # all nine synthetic runs
beacon reproduce ackley-4d --device cpu           # a single run, by tag (see `beacon list`)
beacon reproduce materials --replicates 5         # fewer replicates for a quicker check
```

Results go to `results/generated/paper/<tag>/`. The sensitivity studies in the SI use the same
runs with one setting changed:

```sh
beacon run --problem ackley --dim 8 --set k=5           # k ablation ("Different k")
beacon run --problem ackley --dim 4 --problem-arg n_bins=50   # resolution ("Different Bins")
```

### Single experiments: the scripts in this folder

Each script runs one paper experiment. It is a thin wrapper around `beacon run` with the problem
and algorithm fixed, and it forwards any extra flags:

```sh
python experiments/continuous/single-outcome/ackley-beacon.py --dim 4 --device cuda
python experiments/discrete/single-outcome-material/logd-beacon-saas.py --device cpu --replicates 3
python experiments/discrete/multi-outcome/oil-ug-beacon.py --device auto
```

| Script | Problem | Algorithm |
| --- | --- | --- |
| `continuous/single-outcome/ackley-beacon.py` | `ackley` (default `--dim 12`) | BEACON |
| `continuous/single-outcome/rosenbrock-beacon.py` | `rosenbrock` (default `--dim 12`) | BEACON |
| `continuous/single-outcome/styblinski-tang-beacon.py` | `styblinski-tang` (default `--dim 12`) | BEACON |
| `continuous/single-outcome/noise-ackley-beacon.py` | `noisy-ackley` (default `--noise-std 0.5`) | BEACON |
| `continuous/multi-outcome/synthetic-beacon.py` | `multi-output-plus` | BEACON |
| `continuous/multi-outcome/maze-beacon.py` | `maze` | BEACON |
| `discrete/single-outcome-material/hydrogen-beacon.py` | `hydrogen` | BEACON |
| `discrete/single-outcome-material/nitrogen-beacon.py` | `nitrogen` | BEACON |
| `discrete/single-outcome-material/water-solubility-beacon.py` | `water-solubility` | BEACON |
| `discrete/single-outcome-material/esol-beacon.py` | `esol` | BEACON |
| `discrete/single-outcome-material/logd-beacon.py` | `logd` | BEACON |
| `discrete/single-outcome-material/logd-beacon-saas.py` | `logd` | BEACON-SAAS |
| `discrete/multi-outcome/joint-gas-beacon.py` | `joint-gas` | BEACON |
| `discrete/multi-outcome/oil-beacon.py` | `oil-sorbent` | BEACON |
| `discrete/multi-outcome/oil-ug-beacon.py` | `oil-sorbent` | UG-BEACON |
| `discrete/multi-outcome/mnist/mnist-beacon.py` | `mnist` | BEACON |
| `discrete/multi-outcome/mnist/mnist-ug-beacon.py` | `mnist` | UG-BEACON |

`logd-beacon-saas.py` adds `--set saas_sample_mode=mean` to match the original LogD script, which
averaged posterior samples over the MCMC draws. (The package default draws one proper Thompson
sample from a single MCMC draw.) Script results go to
`results/generated/<problem>/<algorithm>/` unless `--output-dir` is given; the output files are
described under [Outputs](../README.md#outputs).

---

## Benchmark problems

Defaults are the paper settings, verified against the saved results (checked by
[`tests/test_problems.py`](../tests/test_problems.py)). *d* = inputs,
*m* = outcomes, *N* = candidate-set size.

| Problem (`--problem`) | Type | d | m | Initial | Budget | Repl. | k | Reachability regions | Surrogate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `ackley` (`--dim 4/8/12`) | continuous, [-5,5]ᵈ | 4–12 | 1 | 10 | 200 | 20 | 10 | 25 bins on [0, 14.3027] | RBF GP |
| `rosenbrock` | continuous, [-5,5]ᵈ | 4–12 | 1 | 10 | 200 | 20 | 10 | 25 bins on [0, (d−1)·90036] | RBF GP |
| `styblinski-tang` | continuous, [-5,5]ᵈ | 4–12 | 1 | 10 | 200 | 20 | 10 | 25 bins on [−39.166·d, 125·d] | RBF GP |
| `noisy-ackley` (`--noise-std`) | continuous | 4 | 1 | 50 | 300 | 20 | 10 | 50 bins (noise-free values) | RBF GP, learned noise, denoised archive |
| `multi-output-plus` | continuous, [-5,5]⁶ | 6 | 2 | 10 | 500 | 20 | 10 | 10×10 grid on [−5.1, 5.1]² | RBF GPs |
| `maze` | continuous, [-1,1]⁸ | 8 | 2 | ≤10 | 200 | 20 | 10 | metric: best reward | RBF GPs |
| `hydrogen` | discrete, N = 98,694 MOFs | 7 | 1 | 10 | 80 | 20 | 10 | 25 bins (reachable bins) | Matern GP, pathwise sampling |
| `nitrogen` | discrete, N = 5,224 MOFs | 20 | 1 | 10 | 500 | 20 | 10 | 25 bins | Matern GP |
| `joint-gas` | discrete, N = 7,000 MOFs | 25 | 2 | 50 | 500 | 20 | 10 | 10×10 grid | RBF GPs, pathwise |
| `water-solubility` | discrete, N = 900 | 14 | 1 | 10 | 100 | 20 | 10 | 25 bins | Matern GP |
| `esol` | discrete, N = 1,128 | 2,133 (bits) | 1 | 10 | 300 | 20 | 10 | 25 bins | Tanimoto GP |
| `logd` | discrete, N = 1,953 | 125 | 1 | 10 | 100 | 20 | 10 | 25 bins | Matern GP or SAAS |
| `oil-sorbent` | discrete, N = 5,000 (LHS) | 7 | 2 | 10 | 200 | 10 | 3 | 58 non-uniform boxes | RBF GPs, pathwise |
| `mnist` | discrete, N = 2,000 latents | 8 | 196 | 10 | 40 | 10 | 10 | 10 digit classes (CNN p ≥ 0.99) | Higher-order GP |

Short descriptions:

* **Synthetic functions.** Ackley, Rosenbrock and Styblinski-Tang (BoTorch implementations) with
  a scalar outcome, plus *Multi-Output Plus*, a 6-input, 2-outcome function whose outcomes
  concentrate near the centre. In the **noisy Ackley** study, observations carry
  N(0, σ²) noise with σ ∈ {0.5, 1, 2, 4}.
* **Maze.** An 8-parameter linear policy steers a point mass for 300 steps in Gymnasium-Robotics'
  `PointMaze_Large-v3`. The outcome is the final (x, y) position, and the deceptive reward is never
  shown to BEACON.
* **Materials (MOFs).** Hydrogen uptake (Ghude & Chowdhury), nitrogen uptake (Daglar et al.) and
  joint CO₂/CH₄ uptake, each from structural descriptors.
* **Molecules.** Water solubility (Qiao et al.; 14 descriptors), ESOL (fragprint fingerprints with
  a Tanimoto kernel) and chromatographic LogD (125 RDKit descriptors; the BEACON-SAAS study).
* **Oil sorbent** (UG-BEACON). Response surfaces map 7 process variables to oil-absorption capacity
  and mechanical strength. The outcome plane is divided into 58 boxes, finer in the high-capacity
  / high-strength corner.
* **MNIST** (UG-BEACON). Points in an 8-D VAE latent space decode to 14×14 images, and a pretrained
  CNN assigns digit classes. UG-BEACON treats the classes (p ≥ 0.5) as behaviour regions.

---

## Evaluation metrics

**Reachability** (paper Sec. 5) is the fraction of behaviour regions that contain at least one
observed outcome:

`Reach_t = |{φ(yᵢ) : i ≤ t}| / |B|`

* *Scalar outcomes*: `n_bins` equal-width bins over a fixed outcome range. Bins are half-open with
  the last one closed (`numpy.histogram` semantics), and outcomes outside the range count for
  nothing.
* *Two outcomes*: a uniform grid (`GridPartition`) or user-defined boxes (`BoxPartition`).
* *Normalization*: synthetic problems divide by the number of regions. Datasets divide by the
  number of regions occupied by the **whole dataset**, so 1.0 means every reachable region was
  found.
* *Noisy problems*: computed from the noise-free outcomes of the selected inputs.
* *Maze*: the best reward found so far, `1 − ‖final − goal‖ / ‖start − goal‖` (1 = goal reached).
* `beacon.metrics.uniformity` computes the Jensen–Shannon-based uniformity reported by some of the
  original scripts.

The paper reports mean ± one standard deviation over replicates, against the number of
evaluations (the `cost.pt` axis, excluding the initial design).

---

## Reproducing the figures

**Plotting.** [`figures/scripts/*.py`](../figures/scripts/) draw the paper figures from the saved results in
`results/continuous-single-outcome/` and `results/plot-data/`, which include the baselines
(MaxVar, RS, Sobol, NS-EA, NS-DEA, …) that are not part of this package:

```sh
python -m pip install -e ".[plots]"
python figures/scripts/plot-synthetic.py      # -> figures/output/SYNTHETIC.png
```

To plot regenerated BEACON curves against the saved baselines, replace the corresponding tensor.
For example, `results/generated/paper/ackley-4d/coverage.pt` and `cost.pt` correspond to
`results/continuous-single-outcome/4DAckley/4DAckley_coverage_list_BEACON.pt` and
`..._cost_list_BEACON.pt`.

**What to expect.** The code exactly as published with the paper is tagged
[`v1.0.0`](https://github.com/PaulsonLab/BEACON/tree/v1.0.0); this guide describes version 2.0.0.

* Initial designs and metrics match the saved results exactly for the synthetic, noisy Ackley and
  materials/molecule studies (checked by [`tests/test_problems.py`](../tests/test_problems.py)).
* BEACON's later iterations will not match bit-for-bit. Three bugs in the original Thompson
  sampler and evaluation code were fixed, and random streams differ across library versions.
* Final reachability should agree within the replicate spread.
* Joint-gas reachability is the original value × 64/63 (a binning fix).
* The saved oil-sorbent and MNIST plot data came from an earlier script version and cannot be
  regenerated bit-for-bit.

**Compute.** With CPU-only PyTorch on a laptop, an Ackley-4D replicate (200 evaluations) takes about
a minute. Higher dimensions, larger budgets and the dataset problems take longer. BEACON-SAAS
(NUTS) and MNIST (HOGP) are the most expensive and benefit most from a GPU. Run replicates in
parallel by launching several processes with different `--seed` and `--output-dir`.
