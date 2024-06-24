#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 16 14:38:55 2024

@author: tang.1856
"""
import torch
import matplotlib.pyplot as plt

synthetic = '4DAckley'

cost_NS_TS1 = torch.load('/home/tang.1856/Jonathan/Novelty Search/Continuous_Synthetic_Code/Results/Noise/4DAckley/4DNoisyAckley_cost_list_NS_TS_considernoise_1.pt')
coverage_NS_TS1 = torch.load('/home/tang.1856/Jonathan/Novelty Search/Continuous_Synthetic_Code/Results/Noise/4DAckley/4DNoisyAckley_coverage_list_NS_TS_considernoise_1.pt')

cost_NS_TS2 = torch.load('/home/tang.1856/Jonathan/Novelty Search/Continuous_Synthetic_Code/Results/Noise/4DAckley/4DNoisyAckley_cost_list_NS_TS_noise_1.pt')
coverage_NS_TS2 = torch.load('/home/tang.1856/Jonathan/Novelty Search/Continuous_Synthetic_Code/Results/Noise/4DAckley/4DNoisyAckley_coverage_list_NS_TS_noise_1.pt')


coverage_NS_mean_TS1 = torch.mean(coverage_NS_TS1, dim = 0)
coverage_NS_std_TS1 = torch.std(coverage_NS_TS1, dim = 0)
cost_NS_mean_TS1 = torch.mean(cost_NS_TS1, dim = 0)

coverage_NS_mean_TS2 = torch.mean(coverage_NS_TS2, dim = 0)
coverage_NS_std_TS2 = torch.std(coverage_NS_TS2, dim = 0)
cost_NS_mean_TS2 = torch.mean(cost_NS_TS2, dim = 0)


text_size = 24
marker_size = 18
linewidth=4
marker_interval = 15
weight='bold'
alpha = 0.3

plt.figure(figsize=(10,8))
# plt.grid(color='lightgrey', linewidth=0.5)
plt.plot(cost_NS_mean_TS1[::marker_interval], coverage_NS_mean_TS1[::marker_interval], label='BEACON', marker='X', markersize=marker_size, linewidth=linewidth)
plt.plot(cost_NS_mean_TS2[::marker_interval], coverage_NS_mean_TS2[::marker_interval], label='BEACON-noiseless', marker='s', markersize=marker_size, linewidth=linewidth)

plt.fill_between(cost_NS_mean_TS1, coverage_NS_mean_TS1 - coverage_NS_std_TS1, coverage_NS_mean_TS1 + coverage_NS_std_TS1,  alpha=alpha)
plt.fill_between(cost_NS_mean_TS2, coverage_NS_mean_TS2 - coverage_NS_std_TS2, coverage_NS_mean_TS2 + coverage_NS_std_TS2,  alpha=alpha)

plt.xlabel('Number of evaluations', fontsize=text_size, fontweight=weight)
plt.ylabel('Reachability', fontsize=text_size, fontweight=weight)
# plt.legend(prop={'weight':'bold','size':text_size})
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

plt.grid(alpha=0.5, linewidth=2.0)
# plt.title('noise = 0.5', fontsize=text_size)





