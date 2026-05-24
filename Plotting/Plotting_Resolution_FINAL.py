#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Plot reachability sensitivity to grid resolution from saved tensors."""

import sys
from pathlib import Path
import torch
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.paths import CONTINUOUS_SINGLE_OUTCOME_RESULTS_DIR

result_root = CONTINUOUS_SINGLE_OUTCOME_RESULTS_DIR / "Different Bins"

synthetics = ['4DAckley', '4DRosen']
synthetic_names = ['Ackley 4D', 'Rosenbrock 4D']
bins_list = [10, 50, 100]

methods = {
    'BEACON': ('BEACON', 'X', 7),
    'MaxVar': ('MaxVar', '^', 7),
    'NS-EA': ('GA', '>', 1),
    'NS-DEA': ('DEA', 'D', 1),
    'NS-EA-FS': ('NS_x_space', 'p', 7),
    'Sobol': ('sobol', 'o', 7),
    'RS': ('RS', 'v', 7),
}

fig, axes = plt.subplots(2, 3, figsize=(14, 10), dpi=300)

alpha = 0.3
linewidth = 2
marker_size = 5
text_size = 16

for row, synthetic in enumerate(synthetics):
    for col, bins in enumerate(bins_list):
        ax = axes[row, col]

        for label, (file_tag, marker, marker_interval) in methods.items():
            
            cost = torch.load(
                result_root / synthetic / f'{synthetic}_cost_list_{file_tag}_bins{bins}.pt'
            )
            coverage = torch.load(
                result_root / synthetic / f'{synthetic}_coverage_list_{file_tag}_bins{bins}.pt'
            )

            coverage_mean = torch.mean(coverage, dim=0)
            coverage_std = torch.std(coverage, dim=0)
            cost_mean = torch.mean(cost, dim=0)

            ax.plot(
                cost_mean[::marker_interval],
                coverage_mean[::marker_interval],
                label=label,
                marker=marker,
                markersize=marker_size,
                linewidth=linewidth,
            )

            ax.fill_between(
                cost_mean,
                coverage_mean - coverage_std,
                coverage_mean + coverage_std,
                alpha=alpha,
            )

        ax.set_title(
            f'{synthetic_names[row]} (grid={bins})',
            fontsize=text_size,
        )

        ax.grid(alpha=0.5, linewidth=1.0)
        ax.set_ylim(0.05, 1.05)
        ax.tick_params(axis="both", labelsize=text_size)

        if col == 0:
            ax.set_ylabel('Reachability', fontsize=text_size)

        if row == 1:
            ax.set_xlabel('Number of evaluations', fontsize=text_size)

# shared legend
handles, labels = axes[0, 0].get_legend_handles_labels()
fig.legend(
    handles,
    labels,
    loc='lower center',
    bbox_to_anchor=(0.5, 0),
    ncol=7,
    fontsize=16,
)

plt.tight_layout()
plt.subplots_adjust(bottom=0.12)

plt.savefig('DifferentBins_2x3.png', dpi=300, bbox_inches='tight')
# plt.show()
