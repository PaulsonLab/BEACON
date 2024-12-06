#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug  4 15:18:09 2024

@author: tang.1856
"""

import torch
import matplotlib.pyplot as plt
indice = [5,6,7,8,10,11,12,14,15,16] 
synthetic = 'Oil'
cost_BEACON = torch.load('/home/tang.1856/BEACON/BEACON/Discrete_MultiOutcome/Oil_cost_list_BEACON_120.pt')[indice]
coverage_BEACON = torch.load('/home/tang.1856/BEACON/BEACON/Discrete_MultiOutcome/Oil_coverage_list_BEACON_120.pt')[indice]

cost_BEACON_constraint = torch.load('/home/tang.1856/BEACON/BEACON/Discrete_MultiOutcome/Oil_cost_list_BEACON_constraint_120.pt')[indice]
coverage_BEACON_constraint = torch.load('/home/tang.1856/BEACON/BEACON/Discrete_MultiOutcome/Oil_coverage_list_BEACON_constraint_120.pt')[indice]

cost_MaxVar_constraint = torch.load('/home/tang.1856/BEACON/BEACON/Discrete_MultiOutcome/Oil_cost_list_MaxVar_120.pt')[indice]
coverage_MaxVar_constraint = torch.load('/home/tang.1856/BEACON/BEACON/Discrete_MultiOutcome/Oil_coverage_list_MaxVar_120.pt')[indice]

cost_RS_constraint = torch.load('/home/tang.1856/BEACON/BEACON/Discrete_MultiOutcome/Oil_cost_list_RS_120.pt')[indice]
coverage_RS_constraint = torch.load('/home/tang.1856/BEACON/BEACON/Discrete_MultiOutcome/Oil_coverage_list_RS_120.pt')[indice]

# cost_NSFS_constraint = torch.load('/home/tang.1856/BEACON/BEACON/Discrete_MultiOutcome/Oil_cost_list_NS_xspace_120.pt')[indice]
# coverage_NSFS_constraint = torch.load('/home/tang.1856/BEACON/BEACON/Discrete_MultiOutcome/Oil_coverage_list_NS_xspace_120.pt')[indice]


coverage_NS_mean_TS1 = torch.mean(coverage_BEACON, dim = 0)
coverage_NS_std_TS1 = torch.std(coverage_BEACON, dim = 0)
cost_NS_mean_TS1 = torch.mean(cost_BEACON, dim = 0)


coverage_BO_mean = torch.mean(coverage_BEACON_constraint, dim = 0)
coverage_BO_std = torch.std(coverage_BEACON_constraint, dim = 0)
cost_BO_mean = torch.mean(cost_BEACON_constraint, dim = 0)


coverage_MaxVar_mean = torch.mean(coverage_MaxVar_constraint, dim = 0)
coverage_MaxVar_std = torch.std(coverage_MaxVar_constraint, dim = 0)
cost_MaxVar_mean = torch.mean(cost_MaxVar_constraint, dim = 0)

coverage_RS_mean = torch.mean(coverage_RS_constraint, dim = 0)
coverage_RS_std = torch.std(coverage_RS_constraint, dim = 0)
cost_RS_mean = torch.mean(cost_RS_constraint, dim = 0)


text_size = 24
marker_size = 8
weight='bold'
alpha = 0.3
linewidth=1.5
marker_interval = 15
plt.figure(figsize=(3,3), dpi=150)
plt.plot(cost_NS_mean_TS1[::marker_interval], coverage_NS_mean_TS1[::marker_interval], label='BEACON', marker='X', markersize=marker_size,linewidth=linewidth)
plt.plot(cost_BO_mean[::marker_interval], coverage_BO_mean[::marker_interval], label='UG-BEACON', marker='^', markersize=marker_size,linewidth=linewidth)
# plt.plot(cost_NS_xspace_mean[::marker_interval], coverage_NS_xspace_mean[::marker_interval], label='NS-FS', marker='p', markersize=marker_size,color='mediumpurple',linewidth=linewidth)
plt.plot(cost_RS_mean[::marker_interval], coverage_RS_mean[::marker_interval], label='RS', marker='v', markersize=marker_size,color='hotpink',linewidth=linewidth)
plt.plot(cost_MaxVar_mean[::marker_interval], coverage_MaxVar_mean[::marker_interval], label='MaxVar', marker='s', markersize=marker_size,color='green',linewidth=linewidth)

plt.fill_between(cost_NS_mean_TS1, coverage_NS_mean_TS1 - coverage_NS_std_TS1, coverage_NS_mean_TS1 + coverage_NS_std_TS1,  alpha=0.3)
plt.fill_between(cost_BO_mean, coverage_BO_mean - coverage_BO_std, coverage_BO_mean + coverage_BO_std,  alpha=0.3)
# plt.fill_between(cost_NS_xspace_mean, coverage_NS_xspace_mean - coverage_NS_xspace_std, coverage_NS_xspace_mean + coverage_NS_xspace_std,  alpha=0.3,color='mediumpurple')
plt.fill_between(cost_RS_mean, coverage_RS_mean - coverage_RS_std, coverage_RS_mean + coverage_RS_std,  alpha=0.3,color='hotpink')
plt.fill_between(cost_MaxVar_mean, coverage_MaxVar_mean - coverage_MaxVar_std, coverage_MaxVar_mean + coverage_MaxVar_std,  alpha=0.3,color='green')


plt.xlabel('Number of evaluations', fontsize='large')
plt.ylabel('Reachability', fontsize='large')
plt.legend(prop={'weight':'normal','size':'medium'}, loc='best')
# plt.title('7D Electrospun Oil Sorbent', fontsize=text_size, fontweight=weight)

# plt.tick_params(axis='both',
#                 which='both',
#                 width=2)
# ax = plt.gca()
# for label in ax.get_xticklabels():
#     label.set_fontsize(text_size)
#     # label.set_fontweight('bold')
    
# for label in ax.get_yticklabels():
#     label.set_fontsize(text_size)
#     # label.set_fontweight('bold')
# ax.spines['top'].set_linewidth(2)
# ax.spines['bottom'].set_linewidth(2)
# ax.spines['left'].set_linewidth(2)
# ax.spines['right'].set_linewidth(2)
plt.tight_layout()
plt.grid(alpha=0.5, linewidth=2.0)


# from scipy.io import savemat
# save_path = "/home/tang.1856/BEACON/BEACON/Plotting/OilSorbent.mat"
# # Create a dictionary to store all data
# data_dict = {}

# # Iterate through datasets and save data
# for i in range(1):
#     data_dict[f'coverage_BEACON_{i}'] = torch.load('/home/tang.1856/BEACON/BEACON/Discrete_MultiOutcome/Oil_coverage_list_BEACON_120.pt')[indice]
#     data_dict[f'coverage_BEACON_bc_{i}'] = torch.load('/home/tang.1856/BEACON/BEACON/Discrete_MultiOutcome/Oil_coverage_list_BEACON_constraint_120.pt')[indice]
#     data_dict[f'coverage_MaxVar_{i}'] = torch.load('/home/tang.1856/BEACON/BEACON/Discrete_MultiOutcome/Oil_coverage_list_MaxVar_120.pt')[indice]
#     data_dict[f'coverage_RS_{i}'] = torch.load('/home/tang.1856/BEACON/BEACON/Discrete_MultiOutcome/Oil_coverage_list_RS_120.pt')[indice]
    
#     data_dict[f'cost_BEACON_{i}'] = torch.load('/home/tang.1856/BEACON/BEACON/Discrete_MultiOutcome/Oil_cost_list_BEACON_120.pt')[indice]
#     data_dict[f'cost_BEACON_bc_{i}'] = torch.load('/home/tang.1856/BEACON/BEACON/Discrete_MultiOutcome/Oil_cost_list_BEACON_constraint_120.pt')[indice]
#     data_dict[f'cost_MaxVar_{i}'] = torch.load('/home/tang.1856/BEACON/BEACON/Discrete_MultiOutcome/Oil_cost_list_MaxVar_120.pt')[indice]
#     data_dict[f'cost_RS_{i}'] = torch.load('/home/tang.1856/BEACON/BEACON/Discrete_MultiOutcome/Oil_cost_list_RS_120.pt')[indice]
   
# # Save the dictionary to a .mat file
# savemat(save_path, data_dict)



