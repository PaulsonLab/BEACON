#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 16 14:38:55 2024

@author: tang.1856
"""
import torch
import matplotlib.pyplot as plt
from scipy.io import loadmat

synthetic = ['12DAckley','12DRosen','12DStyTang']
synthetic_name = ['Ackley 12D','Rosenbrock 12D','Styblinski-Tang 12D']
bins = 1 
# path = '/home/tang.1856/Jonathan/Novelty Search/Continuous_Synthetic_Code/Results/Different k/'
save_path = "/home/tang.1856/Jonathan/Novelty Search/Continuous_Synthetic_Code/Results/Different k/different_k.mat"
loaded_data = loadmat(save_path)

fig, axes = plt.subplots(3, 1, figsize=(10,12))  # 3 rows, 3 columns
# Add plots and legends to subplots
for i, ax in enumerate(axes.flat):
    # cost_NS_TS1 = torch.load(path+synthetic[i]+'/'+synthetic[i]+'_TS_1_cost_list_NS_'+'k'+str(bins)+'.pt')
    # coverage_NS_TS1 = torch.load(path+synthetic[i]+'/'+synthetic[i]+'_TS_1_coverage_list_NS_'+'k'+str(bins)+'.pt')
    
    # cost_NS_TS2 = torch.load(path+synthetic[i]+'/'+synthetic[i]+'_TS_1_cost_list_NS_'+'k'+str(bins*5)+'.pt')
    # coverage_NS_TS2 = torch.load(path+synthetic[i]+'/'+synthetic[i]+'_TS_1_coverage_list_NS_'+'k'+str(bins*5)+'.pt')
    
    # cost_NS_TS3 = torch.load(path+synthetic[i]+'/'+synthetic[i]+'_TS_1_cost_list_NS_'+'k'+str(bins*10)+'.pt')
    # coverage_NS_TS3 = torch.load(path+synthetic[i]+'/'+synthetic[i]+'_TS_1_coverage_list_NS_'+'k'+str(bins*10)+'.pt')
    
    # cost_NS_TS4 = torch.load(path+synthetic[i]+'/'+synthetic[i]+'_TS_1_cost_list_NS_'+'k'+str(bins*20)+'.pt')
    # coverage_NS_TS4 = torch.load(path+synthetic[i]+'/'+synthetic[i]+'_TS_1_coverage_list_NS_'+'k'+str(bins*20)+'.pt')
    
    cost_NS_TS1 = torch.tensor(loaded_data['cost_NS_TS1_'+str(i)])
    coverage_NS_TS1 =torch.tensor(loaded_data['coverage_NS_TS1_'+str(i)])
    
    cost_NS_TS2 =torch.tensor(loaded_data['cost_NS_TS2_'+str(i)])
    coverage_NS_TS2 = torch.tensor(loaded_data['coverage_NS_TS2_'+str(i)])
    
    cost_NS_TS3 =torch.tensor(loaded_data['cost_NS_TS3_'+str(i)])
    coverage_NS_TS3 =torch.tensor(loaded_data['coverage_NS_TS3_'+str(i)])
    
    cost_NS_TS4 =torch.tensor(loaded_data['cost_NS_TS4_'+str(i)])
    coverage_NS_TS4 =torch.tensor(loaded_data['coverage_NS_TS4_'+str(i)])
    
    
    
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
    
    
    
    marker_interval = 10
    text_size = 14
    marker_size = 10
    weight='bold'
    linewidth = 4
    alpha = 0.3
    # plt.figure(figsize=(16,14))
    
    ax.plot(cost_NS_mean_TS1[::marker_interval], coverage_NS_mean_TS1[::marker_interval], label='k=1', marker='X', markersize=marker_size, linewidth=linewidth)
    ax.fill_between(cost_NS_mean_TS1, coverage_NS_mean_TS1 - coverage_NS_std_TS1, coverage_NS_mean_TS1 + coverage_NS_std_TS1,  alpha=alpha)
    
    ax.plot(cost_NS_mean_TS2[::marker_interval], coverage_NS_mean_TS2[::marker_interval], label='k=5', marker='o', markersize=marker_size, linewidth=linewidth)
    ax.fill_between(cost_NS_mean_TS2, coverage_NS_mean_TS2 - coverage_NS_std_TS2, coverage_NS_mean_TS2 + coverage_NS_std_TS2,  alpha=alpha)
    
    ax.plot(cost_NS_mean_TS3[::marker_interval], coverage_NS_mean_TS3[::marker_interval], label='k=10', marker='^', markersize=marker_size, linewidth=linewidth)
    ax.fill_between(cost_NS_mean_TS3, coverage_NS_mean_TS3 - coverage_NS_std_TS3, coverage_NS_mean_TS3 + coverage_NS_std_TS3,  alpha=alpha)
    
    ax.plot(cost_NS_mean_TS4[::marker_interval], coverage_NS_mean_TS4[::marker_interval], label='k=20', marker='s', markersize=marker_size, linewidth=linewidth)
    ax.fill_between(cost_NS_mean_TS4, coverage_NS_mean_TS4 - coverage_NS_std_TS4, coverage_NS_mean_TS4 + coverage_NS_std_TS4,  alpha=alpha)
    
    if i==2:
        ax.set_xlabel('Number of evaluations', fontsize=text_size)
    ax.set_ylabel('Reachability', fontsize=text_size)
    if i == 0:  # Top-left subplot
       
        ax.legend(loc='lower right', fontsize=text_size)  # Add the legend with custom position and size  

    ax.set_title(synthetic_name[i], fontsize=text_size)  # Add title
    
# plt.subplots_adjust(left=0.2, right=0.95, top=0.95, bottom=0.2, wspace=0.3, hspace=0.3)    
# plt.tick_params(axis='both',
#                 which='both',
#                 width=2)
# ax = plt.gca()
# for label in ax.get_xticklabels():
#     label.set_fontsize(text_size)
#     label.set_fontweight('bold')
    
# for label in ax.get_yticklabels():
#     label.set_fontsize(text_size)
#     label.set_fontweight('bold')
# ax.spines['top'].set_linewidth(2)
# ax.spines['bottom'].set_linewidth(2)
# ax.spines['left'].set_linewidth(2)
# ax.spines['right'].set_linewidth(2)

# plt.grid(alpha=0.5, linewidth=2.0)

# from scipy.io import savemat
# save_path = "/home/tang.1856/Jonathan/Novelty Search/Continuous_Synthetic_Code/Results/Different k/different_k.mat"
# # Create a dictionary to store all data
# data_dict = {}

# # Iterate through datasets and save data
# for i in range(len(synthetic)):
#     data_dict[f'cost_NS_TS1_{i}'] = torch.load(path+synthetic[i]+'/'+synthetic[i]+'_TS_1_cost_list_NS_'+'k'+str(bins)+'.pt')
#     data_dict[f'coverage_NS_TS1_{i}'] = torch.load(path+synthetic[i]+'/'+synthetic[i]+'_TS_1_coverage_list_NS_'+'k'+str(bins)+'.pt')
#     data_dict[f'cost_NS_TS2_{i}'] = torch.load(path+synthetic[i]+'/'+synthetic[i]+'_TS_1_cost_list_NS_'+'k'+str(bins*5)+'.pt')
#     data_dict[f'coverage_NS_TS2_{i}'] = torch.load(path+synthetic[i]+'/'+synthetic[i]+'_TS_1_coverage_list_NS_'+'k'+str(bins*5)+'.pt')
#     data_dict[f'cost_NS_TS3_{i}'] =torch.load(path+synthetic[i]+'/'+synthetic[i]+'_TS_1_cost_list_NS_'+'k'+str(bins*10)+'.pt')
#     data_dict[f'coverage_NS_TS3_{i}'] = torch.load(path+synthetic[i]+'/'+synthetic[i]+'_TS_1_coverage_list_NS_'+'k'+str(bins*10)+'.pt')
#     data_dict[f'cost_NS_TS4_{i}'] = torch.load(path+synthetic[i]+'/'+synthetic[i]+'_TS_1_cost_list_NS_'+'k'+str(bins*20)+'.pt')
#     data_dict[f'coverage_NS_TS4_{i}'] = torch.load(path+synthetic[i]+'/'+synthetic[i]+'_TS_1_coverage_list_NS_'+'k'+str(bins*20)+'.pt')
   

# # Save the dictionary to a .mat file
# savemat(save_path, data_dict)