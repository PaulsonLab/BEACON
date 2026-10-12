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
[Reachability](#measuring-exploration-reachability) ·
[New problems](#using-beacon-on-your-own-problem) ·
[Reproducing the paper](#reproducing-the-paper) ·
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
experiments/               Paper experiments: guide (README.md) + one thin script per experiment
examples/                  Runnable custom-problem examples (continuous oil sorbent, discrete MOF library)
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

### Paper experiments

Each paper experiment also has a ready-made script in [`experiments/`](experiments/); see
[Reproducing the paper](#reproducing-the-paper).

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

## Measuring exploration: reachability

BEACON's progress is measured by **reachability**: the fraction of behaviour regions of outcome
space that contain at least one observed outcome (paper Sec. 5),

`Reach_t = |{φ(yᵢ) : i ≤ t}| / |B|`.

The regions are defined by an *outcome partition* from `beacon.metrics`:

| Partition | Regions | Typical use |
| --- | --- | --- |
| `HistogramPartition(lo, hi, n_bins)` | Equal-width bins of a scalar outcome | One outcome |
| `GridPartition(lower, upper, n_bins)` | Uniform grid over several outcomes | Two (or a few) outcomes |
| `BoxPartition(lower_bounds, upper_bounds)` | Arbitrary boxes, e.g. finer where outcomes are more valuable | Non-uniform resolution |
| `ClassifierPartition(classify, n_classes)` | Predicted class labels | Categorical behaviours (e.g. digits) |

`Reachability(partition)` divides by the number of regions. Not every region may be attainable,
so `Reachability.from_dataset(partition, Y_all)` divides by the regions that a full dataset, or a
dense sample of the design space, actually reaches; 1.0 then means "everything attainable was
found". The same partition defines UG-BEACON's behaviour regions. The settings used for each paper
benchmark are listed in [experiments/README.md](experiments/README.md#evaluation-metrics).

---

## Using BEACON on your own problem

BEACON needs three things from you:

1. a way to **evaluate** inputs: a simulator, a lab experiment, or a lookup in a database;
2. the **design space**: a box of continuous variables, or a finite library of candidates;
3. the **outcomes** whose diversity you care about, as an `(n, m)` tensor.

Two complete, runnable examples built on the paper's materials case studies live in
[`examples/`](examples/). Each first drives BEACON with an ask-tell loop (Part A), then wraps the
same black box as a problem for `run_experiment` (Part B):

```sh
python examples/continuous_oil_sorbent.py --device auto     # continuous design space
python examples/discrete_nitrogen_mofs.py --device auto     # discrete candidate library
# add --quick for a ~10 s smoke test
```

### Example 1 — continuous: designing oil sorbents

Seven process variables (polymer ratios, concentrations, temperatures, …) determine an oil
sorbent's **absorption capacity** and **mechanical strength** (paper Sec. 5.5). The paper's
response surfaces stand in for experiments here. The variables are continuous within physical
bounds, so BEACON maximizes its acquisition with gradients.

**Part A — ask-tell loop.** BEACON proposes; you evaluate:

```python
import torch
from beacon import get_algorithm
from beacon.problems.oil_sorbent import OIL_LOWER, OIL_UPPER, oil_sorbent

LOWER, UPPER = OIL_LOWER.double(), OIL_UPPER.double()     # physical bounds of the 7 variables
SCALE = torch.tensor([2.0e5, 1.25e4], dtype=torch.double)  # typical [capacity, strength]

def sorbent_properties(X):                     # (n, 7) physical -> (n, 2) scaled outcomes
    return oil_sorbent(X).clamp_min(0) / SCALE

def evaluate(x_unit):                          # replace with your simulator or experiment
    return sorbent_properties(LOWER + (UPPER - LOWER) * x_unit)

X = torch.rand(10, 7, dtype=torch.double)      # initial design on the unit cube [0, 1]^7
Y = evaluate(X)
bounds = torch.stack([torch.zeros(7), torch.ones(7)]).double()
algo = get_algorithm("beacon", k=10, kernel="rbf", archive="observed")

for _ in range(30):
    x_next = algo.suggest(X, Y, bounds=bounds)  # (1, 7) in [0, 1]^7
    X, Y = torch.cat([X, x_next]), torch.cat([Y, evaluate(x_next)])

recipes = LOWER + (UPPER - LOWER) * X          # proposed recipes in physical units
```

Two points about this example:

* **Inputs** are optimized on `[0, 1]^d`; map them to physical units inside `evaluate`.
* **Outcomes** are divided by `SCALE` because novelty is a Euclidean distance in outcome space.
  Without it, capacity (~10⁵) would swamp strength (~10⁴). The GP's internal standardization
  (`Standardize`) does not fix this: it only helps each GP fit. Thompson samples and the archive
  are mapped back to the original units before distances are computed.
* This setup differs from the paper's oil-sorbent study (the built-in `oil-sorbent` problem). That
  study uses a fixed 5,000-point candidate set, raw outcome units and k = 3. Here the design space
  is continuous, and the rescaling is a recommended practice rather than the paper's setting.

**Part B — as a problem for the runner.** This adds replicates, seeding, reachability tracking
and result files, and makes UG-BEACON available:

```python
from beacon import ContinuousProblem, run_experiment
from beacon.metrics import GridPartition, Reachability

partition = GridPartition(lower=[0, 0], upper=[1, 1], n_bins=10)   # 10x10 behaviour regions
# Normalize by the cells a dense random sample reaches (1.0 = every reachable cell found).
X_dense = torch.rand(100_000, 7, dtype=torch.double)
reachable = sorbent_properties(LOWER + (UPPER - LOWER) * X_dense)

problem = ContinuousProblem(
    name="oil-sorbent-continuous", objective=sorbent_properties,
    input_dim=7, output_dim=2, lower=LOWER, upper=UPPER,
    metric=Reachability.from_dataset(partition, reachable),
    partition=partition,                                   # also UG-BEACON's regions
    defaults=dict(n_init=10, iterations=100, replicates=3, k=10, kernel="rbf", archive="observed"),
)
for algorithm in ("beacon", "ug-beacon"):
    print(run_experiment(problem, algorithm, device="auto", output_dir=None, verbose=0).summary())
```

After 60 evaluations (2 seeds, CPU), BEACON reaches 66–71% of the 65 reachable cells and UG-BEACON
reaches 75–82%. 70 uniformly random points reach 58–62%.

### Example 2 — discrete: diverse nitrogen uptake in a MOF database

`data/materials/Nitrogen.csv` lists 5,224 metal–organic frameworks with 20 structural and chemical
descriptors and their N₂ uptake (paper Sec. 5.2). Each "evaluation" reveals one MOF's uptake; in
practice that would be a simulation or a measurement. BEACON picks which *untested* MOF to evaluate
next. Use the same pattern for any table of candidates.

**Part A — ask-tell over a candidate library:**

```python
import numpy as np, pandas as pd, torch
from beacon import get_algorithm
from beacon.paths import MATERIALS_DIR

df = pd.read_csv(MATERIALS_DIR / "Nitrogen.csv", encoding="utf-8-sig")
X = torch.tensor(df[df.columns[1:21]].to_numpy(dtype=np.float64))     # 20 descriptors
X = (X - X.min(0).values) / (X.max(0).values - X.min(0).values)       # min-max scale
Y = torch.tensor(df["U_N2 (mol/kg)"].to_numpy(dtype=np.float64)).unsqueeze(-1)

tested = list(np.random.RandomState(0).choice(len(X), size=10, replace=False))  # initial screen
available = torch.ones(len(X), dtype=torch.bool)
available[tested] = False

algo = get_algorithm("beacon", k=10, kernel="matern", archive="observed")   # paper's MOF setup
for _ in range(30):
    idx = algo.suggest(X[tested], Y[tested], candidates=X, available=available)
    # Real campaign: simulate/measure MOF df["MOF"][idx] here and record its uptake.
    tested.append(idx)
    available[idx] = False
```

`suggest` returns the index of the chosen candidate and never repeats one marked unavailable. For
libraries larger than `max_exact_candidates` (6000), Thompson samples switch automatically from
exact joint sampling to the linear-cost pathwise sampler.

**Part B — as a `DiscreteProblem`:**

```python
from beacon import DiscreteProblem, run_experiment
from beacon.metrics import HistogramPartition, Reachability

partition = HistogramPartition(float(Y.min()), float(Y.max()), n_bins=25)   # 25 uptake levels
problem = DiscreteProblem(
    "nitrogen-mofs", X, Y,
    metric=Reachability.from_dataset(partition, Y),   # 1.0 = every attainable level found
    partition=partition,
    defaults=dict(n_init=10, iterations=100, replicates=3, k=10, kernel="matern", archive="observed"),
)
result = run_experiment(problem, "beacon", device="auto")
```

This reproduces the built-in `nitrogen` benchmark exactly (same inputs, outcomes and per-seed
reachability), so it is a good template for your own database. Use `HistogramPartition` or
`GridPartition` for one or two outcomes, `BoxPartition` for non-uniform regions, and
`ClassifierPartition` for class labels. Fingerprint representations can use `kernel="tanimoto"`
(see [Choosing a kernel](#choosing-a-kernel)).

### Practical tips

* **Archive.** Use `archive="observed"` for noise-free or low-noise outcomes. Use
  `"posterior_mean"` (the default) when measurements are noisy, so that noise is not mistaken for
  novelty.
* **Initial design.** About 10 random points (or a space-filling design) is enough to start. The
  paper used 10 for most studies.
* **Making a problem available to the CLI.** `register_problem("name", factory)` makes a problem
  available to `run_experiment("name", ...)` in the same Python process. To make it available to
  `beacon run --problem name`, add it to `_BUILTINS` in `beacon/problems/__init__.py`.

### Extension points

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

## Reproducing the paper

This repository is the companion code of the paper. Every benchmark's paper settings are built in;
they were verified against the saved results, which reproduce exactly for every seed. One command
regenerates a whole study:

```sh
beacon reproduce synthetic --device cuda      # also: noise, materials, molecules, oil-sorbent, mnist, ...
```

**[experiments/README.md](experiments/README.md)** has the full guide: the benchmark problems and
their settings, the per-experiment scripts, the evaluation metrics, regenerating the figures, and
what differs from the original results (see also [CHANGELOG.md](CHANGELOG.md)).

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
