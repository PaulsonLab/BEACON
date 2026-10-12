# Changelog

## 2.0.0 — package refactor (2026-10-11)

> **Looking for the code exactly as published with the paper?** Use the
> [`v1.0.0`](https://github.com/PaulsonLab/BEACON/tree/v1.0.0) tag. Version 2.0.0
> fixes bugs that change BEACON's results (see below), so its numbers will not
> match v1.0.0 bit-for-bit.

This release turns the paper's research scripts into an installable package
(`beacon`) with a single tested implementation of BEACON, UG-BEACON and
BEACON-SAAS, a command-line interface, CPU/GPU device handling and a test
suite. The experiment scripts in `experiments/` still exist under the same
paths but now delegate to the package.

The paper's experimental settings are preserved: they are encoded as problem
defaults (`beacon/problems/`) and were **verified against the tracked results**
that produced the paper figures (see "Verification" below). Algorithmic
behaviour changed only where a bug or an incompatibility was found; each such
change is listed here.

### Bug fixes affecting results

1. **Thompson-sample variance was wrong** (`beacon/thompson_sampling.py`).
   The random Fourier features were scaled by the kernel `outputscale`
   instead of `sqrt(outputscale)`, so prior sample paths had variance
   `outputscale²` instead of `outputscale`. In addition, the pseudo-noise
   in the pathwise update was drawn with the noise *variance* used as the
   standard deviation, and the GP's learned constant mean was ignored.
   *Effect:* samples did not follow the fitted GP. In a check against the
   exact posterior (2-D toy problem, `outputscale ≈ 6`), the std of the
   samples was 2.5× too large far from the data (= `sqrt(outputscale)`).
   Near the data it was off by factors between 0.74× and 2.1×, and far from
   the data the mean reverted to 0 instead of the GP mean. *Fix:* correct scaling, noise std `sqrt(noise)`, constant mean
   included, Cholesky solves instead of an explicit inverse. The sampler now
   matches the exact posterior mean and std (tested for RBF and Matern-½,
   3/2, 5/2 kernels, with and without `ScaleKernel`, and for SAAS models).
   This affects every experiment that used pathwise sampling: all continuous
   problems, joint gas uptake and the oil sorbent.

2. **UG-BEACON could re-select an already-evaluated candidate**
   (`oil-ug-beacon.py`). If every unevaluated candidate's sampled outcome fell
   in an already-visited region, the selection loop ended without choosing,
   and the index from the *previous* iteration was appended again (or a
   `NameError` was raised in the first iteration). *Fix:* fall back to the
   unconstrained BEACON choice among unevaluated candidates; the number of
   fallbacks is recorded in `history.pt`.

3. **Grid reachability counted spurious cells** (joint CO₂/CH₄ uptake). An
   outcome equal to the maximum of one output (but not both) was assigned
   cell index `n_bins`, a zero-width extra cell. The full dataset therefore
   appeared to occupy 64 instead of 63 cells. *Fix:* indices are clamped to
   `n_bins - 1` (`GridPartition`). The visited-cell counts are unchanged (verified
   for all 20 seeds), so reported reachability is the original value
   × 64/63.

4. **Joint gas features were in a non-deterministic order.** Feature columns
   were taken from a Python `set`, whose iteration order changes between
   interpreter runs (hash randomization), so random Fourier features and
   hence results were not reproducible. *Fix:* features are sorted.

5. **Runs were not reproducible.** Only NumPy was seeded (initial design);
   Thompson samples used PyTorch's unseeded RNG. *Fix:* replicate `r` seeds
   Python, NumPy and PyTorch with `seed + r`. The initial designs are
   unchanged.

6. **Maze rollouts treated truncation as success.** The original rollout set
   the final position to the goal when the episode was *terminated or
   truncated*. Only termination (reaching the goal) does this now.
   Truncation cannot happen within the 300-step horizon used in the paper,
   so paper results are unaffected.

### Compatibility fixes

* **pandas ≥ 3:** LogD preprocessing used `df[col].fillna(..., inplace=True)`,
  which is a silent no-op under copy-on-write. NaNs survived and the scripts
  crashed (`torch.histc: range of [nan, nan]`). It is replaced by
  `df.fillna(df.mean())`.
* **BoTorch ≥ 0.12** changed `SingleTaskGP`'s default kernel and likelihood
  (Matern-5/2 + Gamma priors → RBF + LogNormal priors). Models are now always
  built explicitly. `kernel="matern"` with `likelihood="gamma"` reproduces the
  BoTorch 0.10 defaults the paper used, independent of the installed version.
  The SI figure script (`plot-direct-reachability-vs-beacon-1d.py`) relied
  on the default likelihood. Its test failed under BoTorch 0.18 and passes
  again with the likelihood made explicit.
* **BoTorch ≥ 0.12 SAAS models** need JAX + NumPyro, available as the
  `beacon-novelty[saas]` extra; Pyro is still supported for BoTorch ≤ 0.11.
  Newer BoTorch seeds NUTS with `seed=0` on every call, so the seed is now
  drawn from PyTorch's RNG to keep chains different across iterations.
* **Python ≥ 3.12:** `pyDOE2` fails to import (it uses the removed `imp`
  module). It is replaced by a bit-identical Latin hypercube implementation
  (tested against pyDOE2's own code).
* Removed unused or unnecessary dependencies: `scikit-learn`, `pyDOE2`,
  `torchvision` (the MNIST study needs only the bundled VAE/CNN weights; the
  MNIST dataset was downloaded but never used), `linear-operator` (installed
  by GPyTorch). `gauche`, `gymnasium-robotics` and the plotting libraries
  are optional extras.

### Behavioural notes and options

* **BEACON-SAAS Thompson sampling.** The original LogD script averaged 64
  independent posterior draws over the MCMC samples. That approximates the
  posterior mean rather than a Thompson sample, so it removes most of the
  exploration noise. The default (`saas_sample_mode="random"`) now draws from
  one uniformly chosen MCMC sample, which is a proper Thompson sample of the
  hyperparameter-marginalized posterior. `saas_sample_mode="mean"` keeps the
  original behaviour. `beacon reproduce logd-saas` and
  `experiments/.../logd-beacon-saas.py` use it to match the paper run.
* **Denoised archive.** The paper (Eq. 6) measures novelty against the GP
  posterior mean at sampled inputs. The original scripts did this only for
  the noisy Ackley study and used observed outcomes elsewhere. Both are
  available (`archive="posterior_mean" | "observed"`). The library default is
  the paper's `posterior_mean`, and every paper benchmark's defaults keep the
  original script's choice.
* **Continuous BEACON-SAAS** (new): the pathwise sampler supports one MCMC
  hyperparameter sample at a time, so BEACON-SAAS also runs on continuous
  domains with gradient-based acquisition optimization.
* **Tanimoto kernel option** (new): `kernel="tanimoto"` works with BEACON and
  UG-BEACON, on candidate sets and on continuous domains. The formula is
  identical to GAUCHE's kernel (tested to 1e-14), which the ESOL run used.
  Tanimoto has no pathwise sampler, so BEACON uses exact joint sampling: on
  the candidate set, or on a `pool_size` Sobol pool for continuous domains.
  Invalid combinations are rejected when the configuration is built
  (`beacon-saas` or `discrete_sampler="rff"` with Tanimoto), and so are
  negative inputs. `ug_pool_size` was renamed `pool_size` because both uses
  share it.
* **Continuous UG-BEACON** (new): scores a scrambled Sobol pool
  (`pool_size`, default 4096) and picks the best feasible point, because
  the feasibility constraint is not differentiable.
* **Discrete Thompson sampling.** Exact joint sampling is used up to
  `max_exact_candidates` (6000), and the pathwise sampler above that. This
  matters for the 98,694-MOF hydrogen dataset, where an exact joint sample
  needs a 98,694² covariance (~78 GB). The original script noted that it
  might run out of memory.
* **Early stopping.** Once reachability reaches 1, the remaining iterations
  record 1 without spending evaluations (as the original UG oil script
  did). Disable it with `--no-early-stop`.
* The acquisition uses the *sum* of k-NN distances, as the code always did.
  The paper writes the mean; the two have the same maximizer.
* MNIST: the GP now runs in float64 (the original used float32), and
  reachability is reported as a fraction of the 10 digit classes (the paper
  figure plots the count).
* The unused "multiple Thompson samples" option (`TS`, always 1) of the
  discrete scripts was dropped.

### Verification of the preserved paper settings

Run the checks with `pytest tests/test_problems.py`.

* Synthetic problems (Ackley, Rosenbrock, Styblinski-Tang; 4/8/12-D): the
  saved initial reachability is reproduced **exactly for all 20 seeds**.
  This pins `n_init = 10`, the input box [-5, 5]ᵈ, 25 bins and the outcome
  ranges. Saved results have 20 replicates × 200 evaluations.
* Noisy Ackley: `n_init = 50`, 50 bins, 300 evaluations, reachability on
  noise-free values. All reproduced exactly.
* Hydrogen, nitrogen, water solubility, LogD (BEACON and BEACON-SAAS):
  initial reachability is reproduced exactly for all 20 seeds, and the
  reachable-bin normalizers match. Budgets: 80, 500, 100 and 100
  evaluations.
* Joint gas uptake: `n_init = 50`. Counts match exactly; only the
  normalizer changes (bug fix 3).

**Effect of the fixes on performance.** I re-ran BEACON on the 4D synthetics (seeds 0–4,
200 evaluations, CPU) with the corrected sampler. Final reachability matches
the paper's 20-seed results within the replicate spread:

| Problem | After 50 evals (new / paper) | After 200 evals (new / paper) |
| --- | --- | --- |
| Ackley 4D | 0.81 ± 0.07 / 0.83 ± 0.07 | 0.94 ± 0.06 / 0.93 ± 0.07 |
| Rosenbrock 4D | 0.89 ± 0.03 / 0.85 ± 0.06 | 1.00 / 1.00 |
| Styblinski-Tang 4D | 0.86 ± 0.05 / 0.85 ± 0.06 | 1.00 / 0.96 ± 0.02 |

### Discrepancies found (not changed)

* The paper text states 20 initial points for the synthetic benchmarks, but
  the released results were generated with 10 (verified above).
* Several original scripts had `replicate = 1` or `10` and budgets different
  from the saved results (e.g. maze 300 vs. 200 saved; Rosenbrock and
  Styblinski-Tang 1 replicate vs. 20 saved). Problem defaults follow the
  saved results.
* `results/plot-data/OilSorbent.mat` and `MNIST.mat` cannot be regenerated
  bit-for-bit from the released scripts. Their per-seed initial designs
  differ from those the scripts produce (MNIST matches for seeds 0–1 only).
  The candidate set itself matches (the LHS is verified against pyDOE2), as
  does the oil-sorbent normalizer (42 reachable regions), and the mean
  initial oil reachability is identical (8.5 of 42 regions). These files
  were evidently produced by an earlier version of the scripts.
* The original UG script's docstring called UG-BEACON "uncertainty-guided".
  The paper defines it as **user-guided** BEACON (behaviour constraints).

### Structure

* New modules: `device`, `models`, `acquisition`, `algorithms`, `metrics`,
  `problems/`, `runner`, `paper`, `cli`. Shared code replaces 17 copies of
  the acquisition, reachability and model-fitting logic.
* `pyproject.toml` (`pip install -e .`) with optional extras, and a `beacon`
  console script.
* Tests: sampler correctness against the exact posterior, acquisition,
  metrics, algorithm selection, end-to-end CLI/API runs on every available
  device, and checks of the paper settings.
