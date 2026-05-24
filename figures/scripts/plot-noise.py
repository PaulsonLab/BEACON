"""Plot the noisy Ackley sensitivity figure from saved result tensors."""

import sys
from pathlib import Path
import torch
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from beacon.paths import CONTINUOUS_SINGLE_OUTCOME_RESULTS_DIR

noises = [0.5, 1.0, 2.0, 4.0]

text_size = 14
marker_size = 4
linewidth = 2
marker_interval = 15
alpha = 0.3

fig, axes = plt.subplots(2, 2, figsize=(8, 6), dpi=300)

for ax, noise in zip(axes.flat, noises):
    result_dir = CONTINUOUS_SINGLE_OUTCOME_RESULTS_DIR / "Noise" / "4DAckley"

    cost_noise = torch.load(
        result_dir / f'4DNoisyAckley_cost_list_BEACON_considernoise_{noise}.pt'
    )
    coverage_noise = torch.load(
        result_dir / f'4DNoisyAckley_coverage_list_BEACON_considernoise_{noise}.pt'
    )

    cost_noiseless = torch.load(
        result_dir / f'4DNoisyAckley_cost_list_BEACON_considernoiseless_{noise}.pt'
    )
    coverage_noiseless = torch.load(
        result_dir / f'4DNoisyAckley_coverage_list_BEACON_considernoiseless_{noise}.pt'
    )

    cov_noise_mean = torch.mean(coverage_noise, dim=0)
    cov_noise_std = torch.std(coverage_noise, dim=0)
    cost_noise_mean = torch.mean(cost_noise, dim=0)

    cov_noiseless_mean = torch.mean(coverage_noiseless, dim=0)
    cov_noiseless_std = torch.std(coverage_noiseless, dim=0)
    cost_noiseless_mean = torch.mean(cost_noiseless, dim=0)

    ax.plot(
        cost_noise_mean[::marker_interval],
        cov_noise_mean[::marker_interval],
        label='BEACON',
        marker='X',
        markersize=marker_size,
        linewidth=linewidth,
    )
    ax.plot(
        cost_noiseless_mean[::marker_interval],
        cov_noiseless_mean[::marker_interval],
        label='BEACON-noiseless',
        marker='s',
        markersize=marker_size,
        linewidth=linewidth,
    )

    ax.fill_between(
        cost_noise_mean,
        cov_noise_mean - cov_noise_std,
        cov_noise_mean + cov_noise_std,
        alpha=alpha,
    )
    ax.fill_between(
        cost_noiseless_mean,
        cov_noiseless_mean - cov_noiseless_std,
        cov_noiseless_mean + cov_noiseless_std,
        alpha=alpha,
    )

    ax.set_title(rf'$\sigma$={noise}', fontsize=text_size)
    ax.grid(alpha=0.5, linewidth=1.0)

# axis labels only on outer plots
axes[0, 0].set_ylabel('Reachability', fontsize=text_size)
axes[1, 0].set_ylabel('Reachability', fontsize=text_size)
axes[1, 0].set_xlabel('Number of evaluations', fontsize=text_size)
axes[1, 1].set_xlabel('Number of evaluations', fontsize=text_size)

# one shared legend
handles, labels = axes[0, 0].get_legend_handles_labels()
fig.legend(
    handles,
    labels,
    loc='lower center',
    bbox_to_anchor=(0.5, -0.02),
    ncol=2,
    fontsize=12,
)

plt.tight_layout()
plt.subplots_adjust(bottom=0.15)

plt.savefig('NoisyAckley_2x2.png', dpi=300, bbox_inches='tight')
plt.show()
