# BEACON

Companion code for **[BEACON: A Bayesian Optimization Inspired Strategy for Efficient Novelty Search](https://doi.org/10.1039/d6dd00123h)**
(W.-T. Tang, A. Chakrabarty, J. A. Paulson, *Digital Discovery*, 2026; [arXiv:2406.03616](https://arxiv.org/abs/2406.03616)).

BEACON searches for **novel behaviours** in expensive black-box systems: instead of optimizing one
objective, it tries to observe as many *different* outcomes as possible within a small evaluation
budget. It does so by fitting Gaussian-process (GP) surrogates of the outcomes, drawing a
Thompson sample of the outcome function, and evaluating the input whose *predicted* outcome lies
farthest from everything observed so far.

The animation compares random sampling (RS) with BEACON on the paper's synthetic two-outcome
problem. BEACON spreads its samples over the whole outcome space; RS stays in the central region
where most outcomes concentrate (paper, Fig. 3).

![Random sampling vs BEACON](figures/Random_vs_BEACON%20(2).gif)

**Contents**
[Method overview](#method-overview) ·
[Repository structure](#repository-structure) ·
[Installation](#installation) ·
[Quick start](#quick-start) ·
[Running the algorithms](#running-the-algorithms) ·
[Configuration](#configuration) ·
[Benchmarks](#benchmark-problems) ·
[Metrics](#evaluation-metrics) ·
[Reproducing the paper](#reproducing-the-paper) ·
[New problems](#using-beacon-on-your-own-problem) ·
[Testing](#testing) ·
[Changes](#changes-from-the-original-code) ·
[Citation](#citation)

---

## Method overview

### BEACON

Let `f: X → Rᵐ` map design inputs to `m` outcomes (behaviours). Each iteration of BEACON
(Algorithm 1 of the paper):

1. **Model.** Fit an independent GP to each outcome (a block-diagonal multi-output GP).
2. **Sample.** Draw one Thompson sample `g ~ f | D` of the outcome function. For continuous
   domains, BEACON uses the *pathwise* (decoupled) sampler of
   [Wilson et al. (2020)](https://arxiv.org/abs/2002.09309): random Fourier features plus a
   data-dependent correction. This gives a deterministic, differentiable sample path.
3. **Score novelty.** For a candidate `x`, compute the distances from the predicted outcome
   `g(x)` to its `k` nearest neighbours in the archive of observed outcomes:

   `α(x) = Σᵢ₌₁ᵏ ‖g(x) − y₍ᵢ₎‖`  (the paper writes the mean; the maximizer is the same).

   With noisy observations, the archive holds the GP posterior means at the sampled inputs (a
   denoised archive; paper Eq. 6).
4. **Select and evaluate.** Maximize `α`: by multi-start L-BFGS-B over a continuous box, or by
   scoring a finite candidate set (materials and molecule databases). Then evaluate the chosen
   input and repeat.

Thompson sampling supplies the exploration: uncertain regions sometimes get sampled outcomes that
look novel. The k-NN distance in *outcome* space pushes the search toward unexplored
behaviours rather than unexplored inputs.

### Variants

| Algorithm | Selector | What changes | Use it when |
| --- | --- | --- | --- |
| **BEACON** | `beacon` | Exact GPs with MAP hyperparameters (one per outcome). | Default; up to ~20–30 effective input dimensions. |
| **BEACON-SAAS** | `beacon-saas` | Each GP becomes a fully Bayesian GP with a **sparse axis-aligned subspace (SAAS)** prior (Eriksson & Jankowiak, 2021), sampled with NUTS. A Thompson sample picks one MCMC hyperparameter sample, then draws a GP sample under it. | Many inputs of which only a few matter (e.g. 125 molecular descriptors for LogD). Slower per iteration. |
| **UG-BEACON** | `ug-beacon` | **User-guided BEACON** (paper Sec. 4.5): you give a partition of outcome space into behaviour regions. Only inputs whose sampled outcome falls in a region that is *not yet visited* (and not excluded) are eligible. | You care about *which* behaviours are found: finer regions in high-utility parts of outcome space, or classes such as MNIST digits. |

All three share one implementation (`beacon/algorithms.py`). BEACON-SAAS swaps the surrogate and
UG-BEACON swaps the selection rule. Further options, such as kernel choice, a higher-order GP for
image outcomes, or exact vs. pathwise sampling on candidate sets, are configuration settings (see
[Configuration](#configuration)).

For problems with hundreds or thousands of inputs, see the trust-region extension
**TR-BEACON** ([paper](https://openreview.net/pdf?id=9Xo6ONB8E3),
[code](https://github.com/PaulsonLab/TR-BEACON)).

---

## Repository structure

```
beacon/                    The installable package
├── algorithms.py          BEACON, UGBEACON, BEACONSAAS, BeaconConfig, get_algorithm()
├── acquisition.py         k-NN novelty, BeaconAcquisition (continuous), discrete Thompson sampling
├── thompson_sampling.py   PathwiseThompsonSampler (Wilson et al. 2020), RBF/Matern, SAAS-aware
├── models.py              GP construction/fitting: exact GP, SAAS (NUTS), higher-order GP, Tanimoto kernel
├── metrics.py             Outcome partitions (histogram, grid, boxes, classifier) and reachability
├── problems/              Benchmark problems + registry (get_problem, register_problem)
│   ├── synthetic.py       Ackley, Rosenbrock, Styblinski-Tang, noisy Ackley, Multi-Output Plus
│   ├── datasets.py        Hydrogen, nitrogen, joint gas (MOFs); water solubility, ESOL, LogD (molecules)
│   ├── oil_sorbent.py     Oil-sorbent response surfaces, LHS candidate set, non-uniform partition
│   ├── maze.py            Point-maze policy search (Gymnasium-Robotics)
│   └── mnist.py           VAE/CNN MNIST digit discovery
├── runner.py              Replicate loop, seeding, result saving (run_experiment)
├── paper.py               Paper experiment groups for `beacon reproduce`
├── cli.py, __main__.py    Command line: python -m beacon {list,run,reproduce}
├── device.py              Device selection (auto/cpu/cuda), float64, seeding
├── optimization.py        Quiet wrappers around BoTorch's fitting/acquisition optimization
└── paths.py               Repository-relative data/result paths
experiments/               One thin script per paper experiment (calls beacon.cli)
figures/scripts/           Plotting scripts for the paper figures (read results/)
data/materials/            Materials and molecule datasets
models/mnist/              Pretrained VAE and CNN weights for the MNIST study
results/                   Saved paper results (continuous-single-outcome/, plot-data/)
                           and generated outputs (generated/, ignored by git)
tests/                     Unit, integration and paper-setting tests
toc_assets/                Table-of-contents graphic
```

---

## Installation

BEACON needs Python ≥ 3.10, PyTorch ≥ 2.0 and BoTorch ≥ 0.10 (the paper used BoTorch 0.10.0; the
code is tested with BoTorch 0.18).

```sh
git clone https://github.com/PaulsonLab/BEACON.git
cd BEACON
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -e .         # core: torch, botorch, gpytorch, numpy, scipy, pandas
```

**GPU.** Install a CUDA build of PyTorch *before* the step above, following
[pytorch.org](https://pytorch.org/get-started/locally/), for example:

```sh
python -m pip install torch --index-url https://download.pytorch.org/whl/cu124
python -c "import torch; print(torch.cuda.is_available())"
```

**Optional extras.** Install only what you need:

| Extra | Installs | Needed for |
| --- | --- | --- |
| `saas` | `botorch[fully_bayesian]` (JAX + NumPyro) | BEACON-SAAS with BoTorch ≥ 0.12 |
| `saas-legacy` | `pyro-ppl` | BEACON-SAAS with BoTorch ≤ 0.11 |
| `maze` | `gymnasium`, `gymnasium-robotics` | `maze` problem |
| `chem` | `gauche`, `rdkit` | `esol` problem (fingerprints) |
| `plots` | `matplotlib`, `seaborn` | `figures/scripts/` |
| `config` | `pyyaml` | YAML run configurations |
| `dev` | `pytest`, `pyyaml` | running the tests |
| `all` | all of the above except `chem` and `saas-legacy` | |

```sh
python -m pip install -e ".[saas,plots,dev]"
```

`requirements.txt` lists the same core dependencies for `pip install -r requirements.txt`.
The data in `data/` and the model weights in `models/` ship with the repository. Set
`BEACON_DATA_DIR` or `BEACON_RESULTS_DIR` to use other locations.

---

## Quick start

```sh
beacon list                                                       # algorithms, problems, paper groups
beacon run --problem ackley --dim 4 --iterations 50 --replicates 2   # BEACON on 4D Ackley
```

`beacon` and `python -m beacon` are interchangeable. Each run prints per-replicate progress and
writes results to `results/generated/<problem>/<algorithm>/`.

From Python:

```python
from beacon import run_experiment

result = run_experiment("ackley", "beacon", problem_kwargs={"dim": 4},
                        overrides={"iterations": 50, "replicates": 2}, device="auto")
print(result.summary())          # final reachability mean +/- std
result.metric                    # (replicates, iterations + 1) reachability curves
```

---

## Running the algorithms

Choose the algorithm with `--algorithm` (`-a`) and the device with `--device`:

| `--device` | Meaning |
| --- | --- |
| `auto` (default) | CUDA if available, otherwise CPU |
| `cpu` | Force CPU |
| `cuda`, `cuda:1`, … | A specific GPU (fails clearly if CUDA is unavailable) |

Everything runs in float64, which GP fitting needs. Apple `mps` lacks float64 and is rejected
with an explanatory error.

### BEACON

```sh
# Continuous: 8D Ackley, paper settings (10 initial points, 200 evaluations, 20 replicates)
beacon run --problem ackley --dim 8 --algorithm beacon --device cpu
beacon run --problem ackley --dim 8 --algorithm beacon --device cuda

# Discrete candidate set: nitrogen uptake of 5,224 MOFs
beacon run --problem nitrogen --algorithm beacon --device auto --replicates 5

# Two outcomes: joint CO2/CH4 uptake
beacon run --problem joint-gas -a beacon --device cuda
```

### BEACON-SAAS

Install the `saas` extra first. NUTS settings default to the paper's (128 warm-up steps, 64 samples,
thinning 16):

```sh
beacon run --problem logd --algorithm beacon-saas --device cpu
beacon run --problem logd --algorithm beacon-saas --device cuda --set warmup_steps=256 num_samples=128

# Also works on continuous problems (pathwise sample from one MCMC draw):
beacon run --problem ackley --dim 12 --algorithm beacon-saas --iterations 50 --replicates 1
```

### UG-BEACON

UG-BEACON needs behaviour regions. Each benchmark defines its own: the reachability bins or grid
by default, the non-uniform partition for the oil sorbent, and the digit classes for MNIST.

```sh
beacon run --problem oil-sorbent --algorithm ug-beacon --device cpu
beacon run --problem mnist --algorithm ug-beacon --device cuda
# Exclude regions you are not interested in (indices into the partition):
beacon run --problem oil-sorbent -a ug-beacon --set "ug_excluded_regions=[16]"
```

### Experiment scripts

Every paper experiment also has a script under `experiments/`. Each forwards extra flags to
`beacon run`, so the paper defaults can be overridden:

```sh
python experiments/continuous/single-outcome/ackley-beacon.py --dim 4 --device cuda
python experiments/discrete/single-outcome-material/logd-beacon-saas.py --device cpu --replicates 3
python experiments/discrete/multi-outcome/oil-ug-beacon.py --device auto
```

### Outputs

Each run directory contains:

| File | Content |
| --- | --- |
| `coverage.pt` | `(replicates, iterations+1)` reachability after each evaluation (`best_reward.pt` for the maze) |
| `cost.pt` | Matching number of evaluations spent (0 = initial design), the format the plotting scripts use |
| `time.pt` | Seconds per iteration |
| `history.pt` | Per replicate: selected inputs, observed and noise-free outcomes, candidate indices |
| `config.json` | Problem, algorithm, every setting, device and package versions |

Use `--output-dir DIR` to choose the location, `--no-save` to write nothing, and `-v 2` for
per-iteration logging.

---

## Configuration

Settings resolve in this order: **BEACON defaults → problem (paper) defaults → your overrides**.

**Experiment settings** have dedicated flags: `--n-init`, `--iterations`, `--replicates`, `--seed`
(replicate `r` uses seed `seed + r`) and `--no-early-stop`. **Algorithm settings** are passed with
`--set KEY=VALUE`:

| Option | Default | Meaning |
| --- | --- | --- |
| `k` | 10 | Nearest neighbours in the novelty score |
| `archive` | `posterior_mean` | Measure novelty against `observed` outcomes or GP `posterior_mean` (denoised) |
| `surrogate` | `gp` | `gp`, `saas` (set by `beacon-saas`) or `hogp` (tensor outcomes) |
| `kernel` | `rbf` | `rbf`, `matern` (Matern-5/2 + Gamma priors, the BoTorch 0.10 default) or `tanimoto` (fingerprints; see [Choosing a kernel](#choosing-a-kernel)) |
| `likelihood` | `gamma` | `gamma` (Gamma noise prior), `plain` (no prior) or `noiseless` (noise in [1e-6, 1e-4]) |
| `num_bases` | 1024 | Random Fourier features per pathwise sample |
| `discrete_sampler` | `auto` | `exact` joint sampling, pathwise `rff`, or `auto` (exact up to `max_exact_candidates`) |
| `max_exact_candidates` | 6000 | Candidate-set size above which `auto` switches to `rff` |
| `num_restarts`, `raw_samples`, `maxiter`, `batch_limit` | 10, 64, 200, 5 | Continuous acquisition optimization (BoTorch `optimize_acqf`) |
| `warmup_steps`, `num_samples`, `thinning` | 128, 64, 16 | NUTS for BEACON-SAAS |
| `saas_sample_mode` | `random` | `random`: Thompson sample from one MCMC draw; `mean`: average over draws (original LogD script) |
| `pool_size` | 4096 | Sobol pool scored on continuous domains by UG-BEACON, and by BEACON when the kernel has no pathwise sampler (Tanimoto) |
| `ug_excluded_regions` | `[]` | Partition regions UG-BEACON should always avoid |
| `hogp_steps`, `hogp_lr` | 1000, 0.05 | Adam fitting of the higher-order GP (MNIST) |

Problem parameters are set with `--dim`, `--noise-std` or `--problem-arg KEY=VALUE` (for example
`--problem-arg n_bins=50`).

**Config files.** The same settings can come from JSON or YAML; command-line flags take precedence:

```yaml
# my-run.yaml  ->  beacon run --config my-run.yaml --device cuda
problem: styblinski-tang
problem_kwargs: {dim: 8, n_bins: 25}
algorithm: beacon
overrides: {iterations: 100, replicates: 5, k: 5, kernel: matern}
output_dir: results/generated/my-run
```

### Choosing a kernel

| `kernel` | Inputs | Thompson sampling | Typical use |
| --- | --- | --- | --- |
| `rbf` | continuous, scaled to [0, 1]ᵈ | pathwise (gradient-based on continuous domains) | Smooth simulators (continuous paper benchmarks) |
| `matern` | continuous descriptors | pathwise | Rougher responses; materials-descriptor benchmarks |
| `tanimoto` | **non-negative** features: binary or count fingerprints, fragment counts | exact joint sampling | Molecules (ESOL) and other set-like representations |

The **Tanimoto** (Jaccard) kernel `k(x, x') = ⟨x, x'⟩ / (‖x‖² + ‖x'‖² − ⟨x, x'⟩)` measures
fingerprint overlap and has no lengthscales, so it suits high-dimensional sparse bit vectors where
RBF/Matern lengthscales are hard to fit. The implementation matches GAUCHE's kernel used in the
paper, without requiring GAUCHE. Use it with any algorithm except BEACON-SAAS, which has its own
kernel:

```sh
beacon run --problem esol                                    # Tanimoto is ESOL's default
beacon run --problem water-solubility --set kernel=tanimoto  # any dataset with non-negative features
beacon run --problem ackley --dim 4 --set kernel=tanimoto    # continuous: scores a Sobol pool
```

```python
algo = get_algorithm("ug-beacon", kernel="tanimoto", likelihood="plain", partition=part)
idx = algo.suggest(train_fps, train_y, candidates=library_fps, available=mask)
```

Notes:

* Tanimoto is non-stationary, so it has no random-Fourier-feature (pathwise) sampler.
  * **Candidate sets** use exact joint sampling. Its cost grows with N³, and above
    `max_exact_candidates` (6000) BEACON warns about the memory needed. `discrete_sampler=rff` is
    rejected; problems whose preset uses `rff` (joint gas, oil sorbent) also need
    `--set discrete_sampler=exact`.
  * **Continuous domains** draw an exact joint sample on `pool_size` scrambled Sobol points and
    take the most novel one, as UG-BEACON does. Raise `pool_size` for finer resolution.
* Inputs must be non-negative; otherwise BEACON raises an error, because the kernel is not valid
  for negative inputs.

---

## Benchmark problems

Defaults are the paper settings, verified against the saved results (see
[CHANGELOG](CHANGELOG.md#verification-of-the-preserved-paper-settings)). *d* = inputs,
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

## Reproducing the paper

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

**Plotting.** `figures/scripts/*.py` draw the paper figures from the saved results in
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

**What to expect.**

* Initial designs and metrics match the saved results exactly for the synthetic, noisy Ackley and
  materials/molecule studies (checked by `tests/test_problems.py`).
* BEACON's later iterations will not match bit-for-bit. Three bugs in the original Thompson
  sampler and evaluation code were fixed, and random streams differ across library versions.
* Final reachability should agree within the replicate spread. See
  [CHANGELOG.md](CHANGELOG.md) for every change and its effect.
* Joint-gas reachability is the original value × 64/63 (a binning fix).
* The saved oil-sorbent and MNIST plot data came from an earlier script version and cannot be
  regenerated bit-for-bit.

**Compute.** With CPU-only PyTorch on a laptop, an Ackley-4D replicate (200 evaluations) takes about
a minute. Higher dimensions, larger budgets and the dataset problems take longer. BEACON-SAAS
(NUTS) and MNIST (HOGP) are the most expensive and benefit most from a GPU. Run replicates in
parallel by launching several processes with different `--seed` and `--output-dir`.

---

## Using BEACON on your own problem

### 1. Ask-tell use with your own experiments

`suggest` proposes the next input from data you already have. Nothing else is needed:

```python
import torch
from beacon import get_algorithm

algo = get_algorithm("beacon", k=10, archive="posterior_mean")   # or "beacon-saas"

# Continuous design space scaled to [0, 1]^d:
bounds = torch.tensor([[0.0] * 3, [1.0] * 3], dtype=torch.double)
x_next = algo.suggest(train_x, train_y, bounds=bounds)      # (1, d); train_y is (n, m)

# Finite library of candidates (e.g. a materials database):
available = torch.ones(len(X_library), dtype=torch.bool)
available[already_tested] = False
idx = algo.suggest(train_x, train_y, candidates=X_library, available=available)
```

Scale continuous inputs to the unit cube. Outcomes are standardized internally, but they should be
on comparable scales across outputs, because distances are computed in the original outcome units.

### 2. Register a problem to get the runner, CLI, seeding and saving

```python
import torch
from beacon import ContinuousProblem, DiscreteProblem, register_problem, run_experiment
from beacon.metrics import GridPartition, Reachability

def my_simulator(X):                        # X: (n, 3) physical inputs -> (n, 2) outcomes
    return torch.stack([X.sum(-1), (X ** 2).sum(-1)], dim=-1)

def make_my_problem(**kwargs):
    part = GridPartition(lower=[0, 0], upper=[3, 3], n_bins=10)
    return ContinuousProblem(
        name="my-problem", objective=my_simulator, input_dim=3, output_dim=2,
        lower=0.0, upper=1.0,                       # physical box; BEACON works on [0, 1]^d
        metric=Reachability(part), partition=part,  # partition also drives UG-BEACON
        noise_std=0.0,
        defaults={"n_init": 10, "iterations": 100, "replicates": 5, "k": 10, "archive": "observed"},
    )

register_problem("my-problem", make_my_problem, "My simulator")
result = run_experiment("my-problem", "ug-beacon", device="auto")
```

For a dataset, use
`DiscreteProblem(name, X, Y, metric=Reachability.from_dataset(part, Y), partition=part, defaults=...)`.
After `register_problem`, the problem is also available from the CLI within the same Python
process. To make it available to `beacon run`, add an entry to `_BUILTINS` in
`beacon/problems/__init__.py`.

### 3. Extension points

* **Behaviour regions**: subclass `beacon.metrics.OutcomePartition` (implement
  `membership(Y) -> (N, M) bool`), or use `HistogramPartition`, `GridPartition`, `BoxPartition` or
  `ClassifierPartition`.
* **Novelty metric**: `beacon.acquisition.knn_novelty` (Euclidean k-NN). Replace it there to use
  another distance.
* **Surrogates**: `beacon.models` (`make_kernel` adds kernels). Pathwise sampling supports RBF and
  Matern kernels; other kernels (e.g. Tanimoto) automatically use exact joint sampling, on
  candidate sets or on a Sobol pool for continuous domains.
* **New algorithm variants**: subclass `BEACON` and override `fit`, `archive`, `select_candidate`
  or `optimize_continuous`, then add the class to `ALGORITHMS` in `beacon/algorithms.py`.

---

## Testing

```sh
python -m pip install -e ".[dev,saas]"
python -m pytest                       # full suite (~1 min on CPU)
python -m pytest -m "not slow"         # skip SAAS/NUTS tests
```

The suite checks that:

* Thompson samples reproduce the exact GP posterior mean and std (RBF and Matern kernels, SAAS
  models);
* the k-NN acquisition equals the original formulation, and reachability matches the original
  `numpy.histogram` computation;
* algorithm selection, aliases and configuration validation work, along with one step of each
  algorithm on continuous and discrete problems;
* the paper settings reproduce the saved initial reachability for every seed;
* end-to-end API, CLI and experiment-script runs succeed and are reproducible;
* repository hygiene holds.

Device-dependent tests run once per available device. CUDA variants are reported as *skipped* on
machines without a GPU, so run the suite on a GPU machine to cover them.

---

## Changes from the original code

The refactor keeps the paper's experimental settings. It fixes several verified bugs, most
importantly a mis-scaled pathwise Thompson sampler (samples were too wide by a factor of
√outputscale). It also restores compatibility with current pandas, BoTorch and Python, removes
unused dependencies, and replaces 17 near-duplicate scripts with one tested package.
**[CHANGELOG.md](CHANGELOG.md) documents every change, how it was verified, and its effect on
results.**

---

## Citation

```bibtex
@article{tang2026beacon,
  title={BEACON: A Bayesian Optimization Inspired Strategy for Efficient Novelty Search},
  author={Tang, Wei-Ting and Chakrabarty, Ankush and Paulson, Joel A.},
  journal={Digital Discovery},
  year={2026},
  publisher={Royal Society of Chemistry},
  doi={10.1039/d6dd00123h}
}
```

Released under the license in [LICENSE](LICENSE).
