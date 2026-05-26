#!/usr/bin/env python3
"""Illustrate direct reachability and BEACON novelty on a 1D GP example.

This script creates an SI-style figure rather than a benchmark. It compares
the posterior probability of reaching an unreached output bin with BEACON's
archive-distance novelty acquisition under coherent posterior sample paths.
"""

import argparse
import os
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

PLOT_CACHE_DIR = Path(tempfile.gettempdir()) / "beacon-plot-cache"
MPL_CACHE_DIR = PLOT_CACHE_DIR / "matplotlib"
XDG_CACHE_DIR = PLOT_CACHE_DIR / "xdg"
MPL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
XDG_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CACHE_DIR))
os.environ.setdefault("XDG_CACHE_HOME", str(XDG_CACHE_DIR))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
from botorch.models import SingleTaskGP
from botorch.models.transforms.outcome import Standardize
from gpytorch.kernels import RBFKernel, ScaleKernel
from gpytorch.mlls import ExactMarginalLogLikelihood
from matplotlib.patches import Patch
from scipy.stats import spearmanr

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from beacon.optimization import DTYPE, fit_gpytorch_mll_quiet
from beacon.paths import FIGURES_DIR


SEED = 7
SAMPLE_SEED = 23
N_INITIAL = 8
N_GRID = 500
N_BEHAVIOR_BINS = 13
K_NEIGHBORS = 3
N_POSTERIOR_SAMPLES = 128
SIGMA_FLOOR = 1e-9

OUTPUT_STEM = "direct_reachability_vs_beacon_1d"


@dataclass(frozen=True)
class OutputPaths:
    pdf: Path
    png: Path
    caption: Path


@dataclass(frozen=True)
class IllustrationData:
    x_grid: torch.Tensor
    true_y: torch.Tensor
    train_x: torch.Tensor
    train_y: torch.Tensor
    posterior_mean: torch.Tensor
    posterior_sigma: torch.Tensor
    cutpoints: torch.Tensor
    reached_bins: torch.Tensor
    bin_probabilities: torch.Tensor
    direct_reachability: torch.Tensor
    beacon_average: torch.Tensor
    beacon_single: torch.Tensor
    direct_reachability_normalized: torch.Tensor
    beacon_average_normalized: torch.Tensor
    beacon_single_normalized: torch.Tensor
    pearson_correlation: float
    spearman_correlation: float
    two_sigma_coverage: float
    sampling_jitter: float

    @property
    def correlation(self):
        """Backward-compatible alias for the Pearson curve correlation."""
        return self.pearson_correlation


def true_function(x):
    """Smooth multimodal scalar-output function on [0, 1]."""
    return (
        0.75 * torch.sin(2 * torch.pi * x)
        + 0.35 * torch.cos(4 * torch.pi * x + 0.2)
        + 0.15 * torch.sin(7 * torch.pi * x + 0.25)
    )


def normalize_to_unit(values):
    """Normalize a tensor to [0, 1], returning zeros for constant inputs."""
    values = values.detach()
    spread = values.max() - values.min()
    if spread <= 1e-12:
        return torch.zeros_like(values)
    return (values - values.min()) / spread


def curve_correlations(first_curve, second_curve):
    """Return Pearson and Spearman correlations between two 1D curves."""
    pearson = float(torch.corrcoef(torch.stack([first_curve, second_curve]))[0, 1])
    spearman = float(
        spearmanr(
            first_curve.detach().cpu().numpy(),
            second_curve.detach().cpu().numpy(),
        ).correlation
    )
    return pearson, spearman


def interval_coverage(values, lower, upper):
    """Return the fraction of values covered by lower/upper bands."""
    covered = (values >= lower) & (values <= upper)
    return float(covered.to(dtype=DTYPE).mean())


def make_initial_design(seed=SEED, n_initial=N_INITIAL):
    """Create a deterministic sorted initial design."""
    rng = np.random.default_rng(seed)
    train_x = np.sort(rng.uniform(0.04, 0.96, size=n_initial))
    train_x = torch.tensor(train_x[:, None], dtype=DTYPE)
    return train_x, true_function(train_x).to(dtype=DTYPE)


def fit_surrogate(train_x, train_y):
    """Fit the same lightweight GP style used by the continuous examples."""
    covar_module = ScaleKernel(RBFKernel()).to(dtype=DTYPE)
    model = SingleTaskGP(
        train_x,
        train_y,
        outcome_transform=Standardize(m=1),
        covar_module=covar_module,
    )
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_mll_quiet(mll, model=model, seed=SEED, iteration=0)
    model.eval()
    return model


def make_cutpoints(true_y, posterior_mean, posterior_sigma, n_bins=N_BEHAVIOR_BINS):
    """Choose finite behavior-bin cutpoints covering truth and posterior mass."""
    lower = torch.minimum(true_y.min(), (posterior_mean - 3 * posterior_sigma).min())
    upper = torch.maximum(true_y.max(), (posterior_mean + 3 * posterior_sigma).max())
    padding = 0.05 * (upper - lower).clamp_min(1e-12)
    return torch.linspace(lower - padding, upper + padding, n_bins + 1, dtype=DTYPE)[1:-1]


def behavior_bin_probabilities(posterior_mean, posterior_sigma, cutpoints):
    """Return scalar Gaussian probabilities for tail-closed behavior bins."""
    sigma = posterior_sigma.clamp_min(SIGMA_FLOOR)
    lower_edges = torch.cat([torch.tensor([-torch.inf], dtype=DTYPE), cutpoints])
    upper_edges = torch.cat([cutpoints, torch.tensor([torch.inf], dtype=DTYPE)])

    standard_normal = torch.distributions.Normal(
        torch.tensor(0.0, dtype=DTYPE),
        torch.tensor(1.0, dtype=DTYPE),
    )
    upper_z = (upper_edges[None, :] - posterior_mean[:, None]) / sigma[:, None]
    lower_z = (lower_edges[None, :] - posterior_mean[:, None]) / sigma[:, None]
    probabilities = standard_normal.cdf(upper_z) - standard_normal.cdf(lower_z)
    return probabilities.clamp(min=0.0, max=1.0)


def direct_reachability_acquisition(bin_probabilities, reached_bins):
    """Posterior probability of landing in a behavior bin not yet reached."""
    unreached = torch.ones(bin_probabilities.shape[1], dtype=torch.bool)
    unreached[reached_bins] = False
    return bin_probabilities[:, unreached].sum(dim=1).clamp(min=0.0, max=1.0)


def draw_joint_posterior_samples(mean, covariance, n_samples, seed):
    """Draw coherent sample paths over the full grid from the joint posterior."""
    eye = torch.eye(mean.numel(), dtype=mean.dtype, device=mean.device)
    jitter = 1e-8
    for _ in range(8):
        try:
            cholesky = torch.linalg.cholesky(covariance + jitter * eye)
            break
        except RuntimeError:
            jitter *= 10
    else:
        raise RuntimeError("Unable to draw joint posterior samples after adding jitter.")

    generator = torch.Generator(device=mean.device).manual_seed(seed)
    noise = torch.randn(
        n_samples,
        mean.numel(),
        dtype=mean.dtype,
        device=mean.device,
        generator=generator,
    )
    return mean + noise @ cholesky.T, jitter


def beacon_novelty_acquisition(samples, archive_outcomes, k_neighbors=K_NEIGHBORS):
    """Compute scalar-output BEACON archive-distance novelty for sample paths."""
    nearest = min(k_neighbors, archive_outcomes.numel())
    distances = (samples[:, :, None] - archive_outcomes[None, None, :]).abs()
    knn_distances = distances.sort(dim=-1).values[:, :, :nearest]
    return knn_distances.mean(dim=-1)


def build_illustration():
    """Build all deterministic data needed for the figure and tests."""
    torch.manual_seed(SEED)
    train_x, train_y = make_initial_design()
    model = fit_surrogate(train_x, train_y)

    x_grid = torch.linspace(0.0, 1.0, N_GRID, dtype=DTYPE).unsqueeze(-1)
    with torch.no_grad():
        true_y = true_function(x_grid).reshape(-1)
        posterior = model.posterior(x_grid)
        posterior_mean = posterior.mean.reshape(-1)
        posterior_sigma = posterior.variance.clamp_min(1e-12).sqrt().reshape(-1)
        posterior_covariance = posterior.mvn.covariance_matrix.detach().to(dtype=DTYPE)

        cutpoints = make_cutpoints(true_y, posterior_mean, posterior_sigma)
        reached_bins = torch.bucketize(train_y.reshape(-1), cutpoints, right=False).unique()
        bin_probabilities = behavior_bin_probabilities(
            posterior_mean,
            posterior_sigma,
            cutpoints,
        )
        direct_reachability = direct_reachability_acquisition(
            bin_probabilities,
            reached_bins,
        )

        samples, sampling_jitter = draw_joint_posterior_samples(
            posterior_mean,
            posterior_covariance,
            N_POSTERIOR_SAMPLES,
            SAMPLE_SEED,
        )
        archive_outcomes = model.posterior(train_x).mean.reshape(-1)
        beacon_values = beacon_novelty_acquisition(samples, archive_outcomes)
        beacon_average = beacon_values.mean(dim=0)
        beacon_single = beacon_values[0]

    direct_norm = normalize_to_unit(direct_reachability)
    beacon_average_norm = normalize_to_unit(beacon_average)
    beacon_single_norm = normalize_to_unit(beacon_single)
    pearson_correlation, spearman_correlation = curve_correlations(
        direct_norm,
        beacon_average_norm,
    )
    two_sigma_coverage = interval_coverage(
        true_y,
        posterior_mean - 2 * posterior_sigma,
        posterior_mean + 2 * posterior_sigma,
    )

    return IllustrationData(
        x_grid=x_grid.reshape(-1),
        true_y=true_y,
        train_x=train_x.reshape(-1),
        train_y=train_y.reshape(-1),
        posterior_mean=posterior_mean,
        posterior_sigma=posterior_sigma,
        cutpoints=cutpoints,
        reached_bins=reached_bins,
        bin_probabilities=bin_probabilities,
        direct_reachability=direct_reachability,
        beacon_average=beacon_average,
        beacon_single=beacon_single,
        direct_reachability_normalized=direct_norm,
        beacon_average_normalized=beacon_average_norm,
        beacon_single_normalized=beacon_single_norm,
        pearson_correlation=pearson_correlation,
        spearman_correlation=spearman_correlation,
        two_sigma_coverage=two_sigma_coverage,
        sampling_jitter=sampling_jitter,
    )


def output_paths(output_dir):
    output_dir = Path(output_dir)
    return OutputPaths(
        pdf=output_dir / f"{OUTPUT_STEM}.pdf",
        png=output_dir / f"{OUTPUT_STEM}.png",
        caption=output_dir / f"{OUTPUT_STEM}_caption.tex",
    )


def caption_text(spearman_correlation):
    return (
        "\\caption{Illustrative one-dimensional comparison between direct "
        "reachability and BEACON novelty. This example is not a benchmark. "
        "Direct reachability computes the GP posterior probability of landing "
        "in behavior bins that have not yet been reached. BEACON computes "
        "archive-distance novelty using Thompson-sampled posterior outcomes. "
        "The solid BEACON curve averages over posterior samples only to "
        "visualize the typical shape of the sampled acquisition; the algorithm "
        "uses a Thompson-style sampled acquisition. The thin dashed BEACON "
        "curve shows one sampled acquisition path. The two acquisitions are "
        "related but not identical: direct reachability depends on finite "
        "behavior-bin occupancy, whereas BEACON uses continuous archive-distance "
        "novelty in outcome space. For this deterministic "
        f"setup, the normalized direct-reachability and averaged-BEACON curves "
        f"have Spearman rank correlation $\\rho={spearman_correlation:.3f}$, "
        "a rank-based visual diagnostic rather than a performance claim.}\n"
    )


def validate_illustration(data, paths=None):
    """Validate numerical invariants and, optionally, generated files."""
    errors = []

    if not torch.isfinite(data.direct_reachability).all():
        errors.append("direct reachability contains non-finite values")
    if (data.direct_reachability < -1e-8).any() or (data.direct_reachability > 1 + 1e-8).any():
        errors.append("direct reachability is outside [0, 1]")

    probability_sum_error = (data.bin_probabilities.sum(dim=1) - 1).abs().max()
    if probability_sum_error > 1e-6:
        errors.append(f"behavior-bin probabilities do not sum to 1: {probability_sum_error:.3g}")

    for name, values in {
        "averaged BEACON novelty": data.beacon_average,
        "single-sample BEACON novelty": data.beacon_single,
    }.items():
        if not torch.isfinite(values).all():
            errors.append(f"{name} contains non-finite values")
        if (values < -1e-10).any():
            errors.append(f"{name} contains negative values")

    if not np.isfinite(data.pearson_correlation):
        errors.append("Pearson correlation is not finite")
    if not np.isfinite(data.spearman_correlation):
        errors.append("Spearman correlation is not finite")
    if not np.isfinite(data.two_sigma_coverage):
        errors.append("two-sigma coverage is not finite")
    if data.two_sigma_coverage < 0.9:
        errors.append(f"two-sigma coverage is too low: {data.two_sigma_coverage:.3f}")

    if paths is not None:
        for path in (paths.pdf, paths.png, paths.caption):
            if not path.exists() or path.stat().st_size == 0:
                errors.append(f"missing or empty output: {path}")

    if errors:
        raise ValueError("; ".join(errors))


def plot_illustration(data, paths):
    """Write the SI figure to PDF and PNG."""
    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.labelsize": 10,
            "axes.titlesize": 10,
            "legend.fontsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    fig, (top_ax, bottom_ax) = plt.subplots(
        2,
        1,
        figsize=(7.0, 5.85),
        dpi=300,
        sharex=True,
        gridspec_kw={"height_ratios": [1.35, 1.0]},
    )

    band_lower = data.posterior_mean - 2 * data.posterior_sigma
    band_upper = data.posterior_mean + 2 * data.posterior_sigma
    y_min = float(torch.minimum(data.true_y.min(), band_lower.min()))
    y_max = float(torch.maximum(data.true_y.max(), band_upper.max()))
    y_padding = 0.08 * (y_max - y_min)
    y_min -= y_padding
    y_max += y_padding

    lower_edges = torch.cat([torch.tensor([-torch.inf], dtype=DTYPE), data.cutpoints])
    upper_edges = torch.cat([data.cutpoints, torch.tensor([torch.inf], dtype=DTYPE)])
    reached = set(data.reached_bins.tolist())
    reached_bin_color = "#eeeeee"
    unreached_bin_color = "#f6d28b"
    reached_bin_alpha = 0.45
    unreached_bin_alpha = 0.55
    for index, (lower, upper) in enumerate(zip(lower_edges, upper_edges)):
        lower_value = y_min if not torch.isfinite(lower) else max(float(lower), y_min)
        upper_value = y_max if not torch.isfinite(upper) else min(float(upper), y_max)
        if lower_value >= upper_value:
            continue
        if index in reached:
            color = reached_bin_color
            alpha = reached_bin_alpha
        else:
            color = unreached_bin_color
            alpha = unreached_bin_alpha
        top_ax.axhspan(lower_value, upper_value, color=color, alpha=alpha, zorder=0)

    for cutpoint in data.cutpoints:
        top_ax.axhline(
            float(cutpoint),
            color="#8c8c8c",
            linewidth=0.8,
            linestyle=(0, (1.2, 1.8)),
            alpha=0.85,
            zorder=2,
        )

    x_numpy = data.x_grid.cpu().numpy()
    true_line, = top_ax.plot(
        x_numpy,
        data.true_y.cpu().numpy(),
        color="#222222",
        linewidth=2.0,
        label="True function",
    )
    gp_mean_line, = top_ax.plot(
        x_numpy,
        data.posterior_mean.cpu().numpy(),
        color="#1f77b4",
        linewidth=2.0,
        label="GP mean",
    )
    gp_band = top_ax.fill_between(
        x_numpy,
        band_lower.cpu().numpy(),
        band_upper.cpu().numpy(),
        color="#1f77b4",
        alpha=0.18,
        linewidth=0,
        label=r"GP $\pm 2\sigma$",
    )
    observations = top_ax.scatter(
        data.train_x.cpu().numpy(),
        data.train_y.cpu().numpy(),
        color="#111111",
        edgecolor="white",
        linewidth=0.7,
        s=38,
        zorder=5,
        label="Observed points",
    )
    top_ax.set_ylabel("Outcome")
    top_ax.set_ylim(y_min, y_max)
    top_ax.set_title("Surrogate model and behavior regions")
    top_ax.grid(True, color="#e8e8e8", linewidth=0.45, alpha=0.55, zorder=-1)
    top_legend_handles = [
        true_line,
        gp_mean_line,
        gp_band,
        observations,
        Patch(facecolor=reached_bin_color, alpha=reached_bin_alpha, edgecolor="none", label="Reached output bin"),
        Patch(facecolor=unreached_bin_color, alpha=unreached_bin_alpha, edgecolor="none", label="Unreached output bin"),
    ]
    top_ax.legend(
        handles=top_legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.16),
        frameon=False,
        ncol=3,
        columnspacing=1.4,
        handlelength=2.4,
        borderaxespad=0.0,
    )

    reach_x = float(data.x_grid[data.direct_reachability_normalized.argmax()])
    beacon_avg_x = float(data.x_grid[data.beacon_average_normalized.argmax()])
    beacon_single_x = float(data.x_grid[data.beacon_single_normalized.argmax()])

    reach_line, = bottom_ax.plot(
        x_numpy,
        data.direct_reachability_normalized.cpu().numpy(),
        color="#1f77b4",
        linewidth=2.4,
        label="Direct reachability",
    )
    beacon_average_line, = bottom_ax.plot(
        x_numpy,
        data.beacon_average_normalized.cpu().numpy(),
        color="#d95f02",
        linewidth=2.4,
        linestyle="--",
        label="BEACON average",
    )
    beacon_single_line, = bottom_ax.plot(
        x_numpy,
        data.beacon_single_normalized.cpu().numpy(),
        color="#7570b3",
        linewidth=1.3,
        linestyle=(0, (3, 2)),
        label="BEACON sample",
    )

    verticals = [
        (reach_x, "#1f77b4", "Direct max"),
        (beacon_avg_x, "#d95f02", "Avg. BEACON max"),
        (beacon_single_x, "#7570b3", "Sample BEACON max"),
    ]
    vertical_handles = []
    for x_value, color, label in verticals:
        vertical_handles.append(
            bottom_ax.axvline(
                x_value,
                color=color,
                linewidth=1.0,
                linestyle=":",
                alpha=0.85,
                label=label,
            )
        )

    bottom_ax.set_xlabel("Input, $x$")
    bottom_ax.set_ylabel("Acquisition")
    bottom_ax.set_ylim(-0.04, 1.08)
    bottom_ax.set_title("Reachability and BEACON novelty acquisitions")
    bottom_ax.grid(True, color="#e2e2e2", linewidth=0.55, alpha=0.65)
    bottom_ax.legend(
        handles=[reach_line, beacon_average_line, beacon_single_line, *vertical_handles],
        loc="upper center",
        bbox_to_anchor=(0.5, -0.34),
        frameon=False,
        ncol=3,
        columnspacing=1.4,
        handlelength=2.4,
        borderaxespad=0.0,
    )

    fig.subplots_adjust(top=0.82, bottom=0.28, hspace=0.28)
    paths.pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(paths.pdf, bbox_inches="tight")
    fig.savefig(paths.png, dpi=300, bbox_inches="tight")
    plt.close(fig)


def write_outputs(data, output_dir):
    paths = output_paths(output_dir)
    plot_illustration(data, paths)
    paths.caption.write_text(caption_text(data.spearman_correlation), encoding="utf-8")
    return paths


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create a 1D SI figure comparing direct reachability and BEACON novelty.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=FIGURES_DIR,
        help="Directory for PDF, PNG, and caption outputs.",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Validate numerical invariants and generated outputs before exiting.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    data = build_illustration()
    validate_illustration(data)
    paths = write_outputs(data, args.output_dir)
    if args.check:
        validate_illustration(data, paths)

    print(f"Saved PDF to {paths.pdf}")
    print(f"Saved PNG to {paths.png}")
    print(f"Saved caption to {paths.caption}")
    print(f"pearson_correlation={data.pearson_correlation:.3f}")
    print(f"spearman_correlation={data.spearman_correlation:.3f}")
    print(f"two_sigma_coverage={data.two_sigma_coverage:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
