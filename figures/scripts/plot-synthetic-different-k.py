#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Plot a single synthetic k-sensitivity panel from saved tensors."""
import sys
from pathlib import Path
import torch
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from beacon.paths import CONTINUOUS_SINGLE_OUTCOME_RESULTS_DIR

synthetic = '12DAckley'
bins = 1
result_dir = CONTINUOUS_SINGLE_OUTCOME_RESULTS_DIR / "Different k" / synthetic
cost_NS_TS1 = torch.load(result_dir / f"{synthetic}_cost_list_BEACON_k{bins}.pt")
coverage_NS_TS1 = torch.load(result_dir / f"{synthetic}_coverage_list_BEACON_k{bins}.pt")

cost_NS_TS2 = torch.load(result_dir / f"{synthetic}_cost_list_BEACON_k{bins*5}.pt")
coverage_NS_TS2 = torch.load(result_dir / f"{synthetic}_coverage_list_BEACON_k{bins*5}.pt")

cost_NS_TS3 = torch.load(result_dir / f"{synthetic}_cost_list_BEACON_k{bins*10}.pt")
coverage_NS_TS3 = torch.load(result_dir / f"{synthetic}_coverage_list_BEACON_k{bins*10}.pt")

cost_NS_TS4 = torch.load(result_dir / f"{synthetic}_cost_list_BEACON_k{bins*20}.pt")
coverage_NS_TS4 = torch.load(result_dir / f"{synthetic}_coverage_list_BEACON_k{bins*20}.pt")



coverage_NS_mean_TS1 = torch.mean(coverage_NS_TS1, dim = 0)
coverage_NS_std_TS1 = torch.std(coverage_NS_TS1, dim = 0)
cost_NS_mean_TS1 = torch.mean(cost_NS_TS1, dim = 0)

coverage_NS_mean_TS2 = torch.mean(coverage_NS_TS2, dim = 0)
coverage_NS_std_TS2 = torch.std(coverage_NS_TS2, dim = 0)
cost_NS_mean_TS2 = torch.mean(cost_NS_TS2, dim = 0)

coverage_NS_mean_TS3 = torch.mean(coverage_NS_TS3, dim = 0)
coverage_NS_std_TS3 = torch.std(coverage_NS_TS3, dim = 0)
cost_NS_mean_TS3 = torch.mean(cost_NS_TS3, dim = 0)

coverage_NS_mean_TS4 = torch.mean(coverage_NS_TS4, dim = 0)
coverage_NS_std_TS4 = torch.std(coverage_NS_TS4, dim = 0)
cost_NS_mean_TS4 = torch.mean(cost_NS_TS4, dim = 0)



marker_interval = 7
text_size = 24
marker_size = 14
weight='bold'
linewidth = 4
alpha = 0.3
plt.figure(figsize=(16,14))

plt.plot(cost_NS_mean_TS1[::marker_interval], coverage_NS_mean_TS1[::marker_interval], label='k=1', marker='X', markersize=marker_size, linewidth=linewidth)
plt.fill_between(cost_NS_mean_TS1, coverage_NS_mean_TS1 - coverage_NS_std_TS1, coverage_NS_mean_TS1 + coverage_NS_std_TS1,  alpha=alpha)

plt.plot(cost_NS_mean_TS2[::marker_interval], coverage_NS_mean_TS2[::marker_interval], label='k=5', marker='o', markersize=marker_size, linewidth=linewidth)
plt.fill_between(cost_NS_mean_TS2, coverage_NS_mean_TS2 - coverage_NS_std_TS2, coverage_NS_mean_TS2 + coverage_NS_std_TS2,  alpha=alpha)

plt.plot(cost_NS_mean_TS3[::marker_interval], coverage_NS_mean_TS3[::marker_interval], label='k=10', marker='^', markersize=marker_size, linewidth=linewidth)
plt.fill_between(cost_NS_mean_TS3, coverage_NS_mean_TS3 - coverage_NS_std_TS3, coverage_NS_mean_TS3 + coverage_NS_std_TS3,  alpha=alpha)

plt.plot(cost_NS_mean_TS4[::marker_interval], coverage_NS_mean_TS4[::marker_interval], label='k=20', marker='s', markersize=marker_size, linewidth=linewidth)
plt.fill_between(cost_NS_mean_TS4, coverage_NS_mean_TS4 - coverage_NS_std_TS4, coverage_NS_mean_TS4 + coverage_NS_std_TS4,  alpha=alpha)

plt.xlabel('Number of evaluations', fontsize=text_size, fontweight=weight)
plt.ylabel('Reachability', fontsize=text_size, fontweight=weight)
# plt.legend(prop={'weight':'bold','size':text_size}, loc=2)
# plt.title(synthetic)

plt.tick_params(axis='both',
                which='both',
                width=2)
ax = plt.gca()
for label in ax.get_xticklabels():
    label.set_fontsize(text_size)
    label.set_fontweight('bold')
    
for label in ax.get_yticklabels():
    label.set_fontsize(text_size)
    label.set_fontweight('bold')
ax.spines['top'].set_linewidth(2)
ax.spines['bottom'].set_linewidth(2)
ax.spines['left'].set_linewidth(2)
ax.spines['right'].set_linewidth(2)
plt.ylim(0,1)
plt.grid(alpha=0.5, linewidth=2.0)
