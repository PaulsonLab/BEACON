#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Plot material-discovery benchmark panels from saved figure inputs."""
import sys
from pathlib import Path
import torch
import matplotlib.pyplot as plt
from scipy.io import loadmat

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.paths import FIGURE_INPUTS_DIR

synthetic = ['H2','N2uptake', 'MOF']
synthetic_name = ['Hydrogen uptake capacity','Nitrogen uptake capacity', 'Joint gas uptake capacity']

text_size = 18
marker_size = 8
linewidth=4

weight='bold'
alpha = 0.3

save_path = FIGURE_INPUTS_DIR / "MOF.mat"
loaded_data = loadmat(save_path)


fig, axes = plt.subplots(1, 3, figsize=(12, 6))  # 3 rows, 3 columns
# Add plots and legends to subplots
for i, ax in enumerate(axes.flat):
    
    if i==0:
        marker_interval = 5
    else:
        marker_interval = 25
    
    cost_NS_TS1 = torch.tensor(loaded_data['cost_NS_TS1_'+str(i)])
    coverage_NS_TS1 = torch.tensor(loaded_data['coverage_NS_TS1_'+str(i)])

    cost_BO = torch.tensor(loaded_data['cost_BO_'+str(i)])
    coverage_BO = torch.tensor(loaded_data['coverage_BO_'+str(i)])

    cost_RS = torch.tensor(loaded_data['cost_RS_'+str(i)])
    coverage_RS = torch.tensor(loaded_data['coverage_RS_'+str(i)])

    coverage_NS_mean_TS1 = torch.mean(coverage_NS_TS1, dim = 0)
    coverage_NS_std_TS1 = torch.std(coverage_NS_TS1, dim = 0)
    cost_NS_mean_TS1 = torch.mean(cost_NS_TS1, dim = 0)

    coverage_BO_mean = torch.mean(coverage_BO, dim = 0)
    coverage_BO_std = torch.std(coverage_BO, dim = 0)
    cost_BO_mean = torch.mean(cost_BO, dim = 0)

    coverage_RS_mean = torch.mean(coverage_RS, dim = 0)
    coverage_RS_std = torch.std(coverage_RS, dim = 0)
    cost_RS_mean = torch.mean(cost_RS, dim = 0)

    # coverage_NS_mean_mean = torch.mean(coverage_NS_mean, dim = 0)
    # coverage_NS_mean_std = torch.std(coverage_NS_mean, dim = 0)
    # cost_NS_mean_mean = torch.mean(cost_NS_mean, dim = 0)
   
    ax.plot(cost_NS_mean_TS1[::marker_interval], coverage_NS_mean_TS1[::marker_interval], label='BEACON', marker='X', markersize=marker_size, linewidth=linewidth)
    ax.plot(cost_BO_mean[::marker_interval], coverage_BO_mean[::marker_interval], label='MaxVar', marker='^', markersize=marker_size, linewidth=linewidth)  
    ax.plot(cost_RS_mean[::marker_interval], coverage_RS_mean[::marker_interval], label='RS', marker='v', markersize=marker_size, linewidth=linewidth )
    ax.fill_between(cost_NS_mean_TS1, coverage_NS_mean_TS1 - coverage_NS_std_TS1, coverage_NS_mean_TS1 + coverage_NS_std_TS1,  alpha=alpha)
    ax.fill_between(cost_BO_mean, coverage_BO_mean - coverage_BO_std, coverage_BO_mean + coverage_BO_std,  alpha=alpha) 
    ax.fill_between(cost_RS_mean, coverage_RS_mean - coverage_RS_std, coverage_RS_mean + coverage_RS_std,  alpha=alpha)

    ax.set_ylim(0.15,1.05)

    ax.set_title(synthetic_name[i], fontsize=text_size)  # Add title
    ax.grid(True)  # Add grid
    
    # Add a legend to the top-left subplot (subplot 0)
    if i == 0:  # Top-left subplot
        ax.set_ylabel('Reachability', fontsize=text_size)
    # if i==2:
    #     ax.legend(loc='lower right', fontsize=text_size)  # Add the legend with custom position and size
    # if i==3 or i==6:
    #     ax.set_ylabel('Reachability', fontsize=14)
    
    # if i==7 or i==8 or i==6:
    #     ax.set_xlabel('Number of evaluations', fontsize=14)
    ax.set_xlabel('Number of evaluations', fontsize=text_size)
    ax.tick_params(axis="both", labelsize=text_size)
         
# Get handles and labels from first subplot
handles, labels = axes[0].get_legend_handles_labels()

# Shared legend at bottom
fig.legend(
    handles,
    labels,
    loc='lower center',
    bbox_to_anchor=(0.5, -0.01),
    ncol=3,
    fontsize=text_size
)

# Leave space for legend
plt.subplots_adjust(left=0.08, right=0.99,bottom=0.22, wspace=0.25)

# leave space for legend
# plt.subplots_adjust(bottom=0.18, wspace=0.25)
  
# plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05, wspace=0.2, hspace=0.2)        
plt.savefig('MOF.png',dpi=300)


