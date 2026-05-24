#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Plot the continuous synthetic benchmark figure from saved tensors."""
import sys
from pathlib import Path
import torch
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.paths import CONTINUOUS_SINGLE_OUTCOME_RESULTS_DIR

synthetic = ['4DAckley','4DRosen','4DStyTang','8DAckley','8DRosen','8DStyTang','12DAckley','12DRosen','12DStyTang']
synthetic_name = ['Ackley 4D','Rosenbrock 4D','Styblinski-Tang 4D','Ackley 8D','Rosenbrock 8D','Styblinski-Tang 8D','Ackley 12D','Rosenbrock 12D','Styblinski-Tang 12D']

text_size = 18
marker_size = 10
linewidth=4
marker_interval = 10
weight='bold'
alpha = 0.3


fig, axes = plt.subplots(3, 3, figsize=(16, 14), dpi=300)  # 3 rows, 3 columns
# Add plots and legends to subplots
for i, ax in enumerate(axes.flat):
    result_dir = CONTINUOUS_SINGLE_OUTCOME_RESULTS_DIR / synthetic[i]
    
    cost_NS_TS1 = torch.load(result_dir / f"{synthetic[i]}_cost_list_BEACON.pt")
    coverage_NS_TS1 = torch.load(result_dir / f"{synthetic[i]}_coverage_list_BEACON.pt")

    cost_BO = torch.load(result_dir / f"{synthetic[i]}_cost_list_MaxVar.pt")
    coverage_BO = torch.load(result_dir / f"{synthetic[i]}_coverage_list_MaxVar.pt")

    cost_RS = torch.load(result_dir / f"{synthetic[i]}_cost_list_RS.pt")
    coverage_RS = torch.load(result_dir / f"{synthetic[i]}_coverage_list_RS.pt")

    cost_NS_mean = torch.load(result_dir / f"{synthetic[i]}_cost_list_NS_mean.pt")
    coverage_NS_mean = torch.load(result_dir / f"{synthetic[i]}_coverage_list_NS_mean.pt")

    cost_GA_NS_random = torch.load(result_dir / f"{synthetic[i]}_cost_list_GA_random.pt")
    coverage_GA_NS_random = torch.load(result_dir / f"{synthetic[i]}_coverage_list_GA_random.pt")

    cost_GA_NS_novel = torch.load(result_dir / f"{synthetic[i]}_cost_list_GA_novel.pt")
    coverage_GA_NS_novel = torch.load(result_dir / f"{synthetic[i]}_coverage_list_GA_novel.pt")

    cost_DEA = torch.load(result_dir / f"{synthetic[i]}_cost_list_GA_DEA.pt")
    coverage_DEA = torch.load(result_dir / f"{synthetic[i]}_coverage_list_GA_DEA.pt")

    cost_sobol = torch.load(result_dir / f"{synthetic[i]}_cost_list_sobol.pt")
    coverage_sobol = torch.load(result_dir / f"{synthetic[i]}_coverage_list_sobol.pt")

    cost_NS_xspace = torch.load(result_dir / f"{synthetic[i]}_cost_list_NS_x_space.pt")
    coverage_NS_xspace = torch.load(result_dir / f"{synthetic[i]}_coverage_list_NS_x_space.pt")

    coverage_NS_mean_TS1 = torch.mean(coverage_NS_TS1, dim = 0)
    coverage_NS_std_TS1 = torch.std(coverage_NS_TS1, dim = 0)
    cost_NS_mean_TS1 = torch.mean(cost_NS_TS1, dim = 0)

    coverage_BO_mean = torch.mean(coverage_BO, dim = 0)
    coverage_BO_std = torch.std(coverage_BO, dim = 0)
    cost_BO_mean = torch.mean(cost_BO, dim = 0)

    coverage_RS_mean = torch.mean(coverage_RS, dim = 0)
    coverage_RS_std = torch.std(coverage_RS, dim = 0)
    cost_RS_mean = torch.mean(cost_RS, dim = 0)

    coverage_NS_mean_mean = torch.mean(coverage_NS_mean, dim = 0)
    coverage_NS_mean_std = torch.std(coverage_NS_mean, dim = 0)
    cost_NS_mean_mean = torch.mean(cost_NS_mean, dim = 0)

    coverage_GA_NS_random_mean = torch.mean(coverage_GA_NS_random , dim = 0)
    coverage_GA_NS_random_std = torch.std(coverage_GA_NS_random , dim = 0)
    cost_GA_NS_random_mean = torch.mean(cost_GA_NS_random , dim = 0)

    coverage_GA_NS_novel_mean = torch.mean(coverage_GA_NS_novel, dim = 0)
    coverage_GA_NS_novel_std = torch.std(coverage_GA_NS_novel , dim = 0)
    cost_GA_NS_novel_mean = torch.mean(cost_GA_NS_novel , dim = 0)

    coverage_DEA_mean = torch.mean(coverage_DEA, dim = 0)
    coverage_DEA_std = torch.std(coverage_DEA , dim = 0)
    cost_DEA_mean = torch.mean(cost_DEA , dim = 0)

    coverage_sobol_mean = torch.mean(coverage_sobol, dim = 0)
    coverage_sobol_std = torch.std(coverage_sobol , dim = 0)
    cost_sobol_mean = torch.mean(cost_sobol , dim = 0)

    coverage_NS_xspace_mean = torch.mean(coverage_NS_xspace, dim = 0)
    coverage_NS_xspace_std = torch.std(coverage_NS_xspace , dim = 0)
    cost_NS_xspace_mean = torch.mean(cost_NS_xspace , dim = 0)

    # coverage_EI_mean = torch.mean(coverage_EI, dim = 0)
    # coverage_EI_std = torch.std(coverage_EI , dim = 0)
    # cost_EI_mean = torch.mean(cost_EI , dim = 0)
    
    
    ax.plot(cost_NS_mean_TS1[::marker_interval], coverage_NS_mean_TS1[::marker_interval], label='BEACON', marker='X', markersize=marker_size, linewidth=linewidth)
    ax.plot(cost_BO_mean[::marker_interval], coverage_BO_mean[::marker_interval], label='MaxVar', marker='^', markersize=marker_size, linewidth=linewidth)
    ax.plot(cost_GA_NS_novel_mean, coverage_GA_NS_novel_mean, label='NS-EA', marker='>', markersize=marker_size, linewidth=linewidth )
    ax.plot(cost_DEA_mean, coverage_DEA_mean, label='NS-DEA', marker='D', markersize=marker_size, linewidth=linewidth )
    # ax.plot(cost_NS_xspace_mean[::marker_interval], coverage_NS_xspace_mean[::marker_interval], label='NS-FS', marker='p', markersize=marker_size, linewidth=linewidth )
    ax.plot(cost_sobol_mean[::marker_interval], coverage_sobol_mean[::marker_interval], label='Sobol', marker='o', markersize=marker_size, linewidth=linewidth )
    ax.plot(cost_RS_mean[::marker_interval], coverage_RS_mean[::marker_interval], label='RS', marker='v', markersize=marker_size, linewidth=linewidth )
    # ax.plot(cost_EI_mean[::marker_interval], coverage_EI_mean[::marker_interval], label='EI', marker='*', markersize=marker_size, linewidth=linewidth )

    ax.fill_between(cost_NS_mean_TS1, coverage_NS_mean_TS1 - coverage_NS_std_TS1, coverage_NS_mean_TS1 + coverage_NS_std_TS1,  alpha=alpha)
    ax.fill_between(cost_BO_mean, coverage_BO_mean - coverage_BO_std, coverage_BO_mean + coverage_BO_std,  alpha=alpha)
    ax.fill_between(cost_GA_NS_novel_mean, coverage_GA_NS_novel_mean - coverage_GA_NS_novel_std, coverage_GA_NS_novel_mean + coverage_GA_NS_novel_std,  alpha=alpha)
    ax.fill_between(cost_DEA_mean, coverage_DEA_mean - coverage_DEA_std, coverage_DEA_mean + coverage_DEA_std,  alpha=alpha)
    # ax.fill_between(cost_NS_xspace_mean, coverage_NS_xspace_mean - coverage_NS_xspace_std, coverage_NS_xspace_mean + coverage_NS_xspace_std,  alpha=alpha)
    ax.fill_between(cost_sobol_mean, coverage_sobol_mean - coverage_sobol_std, coverage_sobol_mean + coverage_sobol_std,  alpha=alpha)
    ax.fill_between(cost_RS_mean, coverage_RS_mean - coverage_RS_std, coverage_RS_mean + coverage_RS_std,  alpha=alpha)
    # ax.fill_between(cost_EI_mean, coverage_EI_mean - coverage_EI_std, coverage_EI_mean + coverage_EI_std,  alpha=alpha)
    
    ax.set_ylim(0.15,1.05)

    ax.set_title(synthetic_name[i], fontsize=text_size)  # Add title
    ax.grid(True)  # Add grid
    ax.tick_params(axis="both", labelsize=text_size)
    
    # Add a legend to the top-left subplot (subplot 0)
    if i == 0:  # Top-left subplot
        ax.set_ylabel('Reachability', fontsize=text_size)
        # ax.legend(loc='upper left', fontsize=12)  # Add the legend with custom position and size
    if i==3 or i==6:
        ax.set_ylabel('Reachability', fontsize=text_size)
    
    if i==7 or i==8 or i==6:
        ax.set_xlabel('Number of evaluations', fontsize=text_size)

# Get handles and labels from one subplot
handles, labels = axes[0,0].get_legend_handles_labels()

# Put shared legend at bottom
fig.legend(
    handles,
    labels,
    loc='lower center',
    bbox_to_anchor=(0.5, 0.02),
    ncol=6,
    fontsize=text_size
)

# Leave space for legend
plt.subplots_adjust(
    left=0.05,
    right=0.95,
    top=0.95,
    bottom=0.12,   # increase bottom margin
    wspace=0.2,
    hspace=0.2
)    
fig.set_constrained_layout_pads(w_pad=0.02, h_pad=0.02, wspace=0.02, hspace=0.02)
# plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05, wspace=0.2, hspace=0.2)        
plt.savefig('SYNTHETIC.png', dpi=300)


