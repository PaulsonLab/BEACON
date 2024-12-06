#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 16 14:38:55 2024

@author: tang.1856
"""
import torch
import matplotlib.pyplot as plt
from scipy.io import loadmat

synthetic = ['H2','N2uptake', 'MOF']
synthetic_name = ['Hydrogen uptake capacity','Nitrogen uptake capacity', 'Joint gas uptake capacity']

text_size = 16
marker_size = 8
linewidth=4

weight='bold'
alpha = 0.3

save_path = "/home/tang.1856/BEACON/BEACON/Plotting/MOF.mat"
loaded_data = loadmat(save_path)


fig, axes = plt.subplots(1, 3, figsize=(12, 8))  # 3 rows, 3 columns
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

    # cost_NS_mean = torch.load('/home/tang.1856/Jonathan/Novelty Search/Continuous_Synthetic_Code/Results/'+synthetic[i]+'/'+synthetic[i]+'_cost_list_NS_mean.pt')
    # coverage_NS_mean = torch.load('/home/tang.1856/Jonathan/Novelty Search/Continuous_Synthetic_Code/Results/'+synthetic[i]+'/'+synthetic[i]+'_coverage_list_NS_mean.pt')


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
        ax.legend(loc='upper left', fontsize=text_size)  # Add the legend with custom position and size
    # if i==3 or i==6:
    #     ax.set_ylabel('Reachability', fontsize=14)
    
    # if i==7 or i==8 or i==6:
    #     ax.set_xlabel('Number of evaluations', fontsize=14)
    ax.set_xlabel('Number of evaluations', fontsize=text_size)
        
# plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05, wspace=0.2, hspace=0.2)        




