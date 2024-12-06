#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 16 14:38:55 2024

@author: tang.1856
"""
import torch
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy.io import savemat

synthetic = '4DAckley'
noise = [0.5,1,2,4]
synthetic_name = ['\u03C3=0.5','\u03C3=1.0','\u03C3=2.0','\u03C3=4.0']

save_path = "/home/tang.1856/Jonathan/Novelty Search/Continuous_Synthetic_Code/Results/Noise/4DAckley/noise.mat"
loaded_data = loadmat(save_path)

fig, axes = plt.subplots(2, 2, figsize=(14, 14)) 
# Add plots and legends to subplots
for i, ax in enumerate(axes.flat):
    
    # cost_NS_TS1 = torch.load('/home/tang.1856/Jonathan/Novelty Search/Continuous_Synthetic_Code/Results/Noise/4DAckley/4DNoisyAckley_cost_list_NS_TS_considernoise_'+str(noise[i])+'.pt')
    # coverage_NS_TS1 = torch.load('/home/tang.1856/Jonathan/Novelty Search/Continuous_Synthetic_Code/Results/Noise/4DAckley/4DNoisyAckley_coverage_list_NS_TS_considernoise_'+str(noise[i])+'.pt')
    
    # cost_NS_TS2 = torch.load('/home/tang.1856/Jonathan/Novelty Search/Continuous_Synthetic_Code/Results/Noise/4DAckley/4DNoisyAckley_cost_list_NS_TS_noise_'+str(noise[i])+'.pt')
    # coverage_NS_TS2 = torch.load('/home/tang.1856/Jonathan/Novelty Search/Continuous_Synthetic_Code/Results/Noise/4DAckley/4DNoisyAckley_coverage_list_NS_TS_noise_'+str(noise[i])+'.pt')
    
    cost_NS_TS1 = torch.tensor(loaded_data['cost_NS_TS1_'+str(i)])
    coverage_NS_TS1 = torch.tensor(loaded_data['coverage_NS_TS1_'+str(i)])
    
    cost_NS_TS2 = torch.tensor(loaded_data['cost_NS_TS2_'+str(i)])
    coverage_NS_TS2 = torch.tensor(loaded_data['coverage_NS_TS2_'+str(i)])
    
    
    coverage_NS_mean_TS1 = torch.mean(coverage_NS_TS1, dim = 0)
    coverage_NS_std_TS1 = torch.std(coverage_NS_TS1, dim = 0)
    cost_NS_mean_TS1 = torch.mean(cost_NS_TS1, dim = 0)
    
    coverage_NS_mean_TS2 = torch.mean(coverage_NS_TS2, dim = 0)
    coverage_NS_std_TS2 = torch.std(coverage_NS_TS2, dim = 0)
    cost_NS_mean_TS2 = torch.mean(cost_NS_TS2, dim = 0)
    
    
    text_size = 22
    marker_size = 10
    linewidth=4
    marker_interval = 30
    weight='bold'
    alpha = 0.3
    
    # plt.figure(figsize=(10,8))
    ax.plot(cost_NS_mean_TS1[::marker_interval], coverage_NS_mean_TS1[::marker_interval], label='BEACON', marker='X', markersize=marker_size, linewidth=linewidth)
    ax.plot(cost_NS_mean_TS2[::marker_interval], coverage_NS_mean_TS2[::marker_interval], label='BEACON-noiseless', marker='s', markersize=marker_size, linewidth=linewidth)
    
    ax.fill_between(cost_NS_mean_TS1, coverage_NS_mean_TS1 - coverage_NS_std_TS1, coverage_NS_mean_TS1 + coverage_NS_std_TS1,  alpha=alpha)
    ax.fill_between(cost_NS_mean_TS2, coverage_NS_mean_TS2 - coverage_NS_std_TS2, coverage_NS_mean_TS2 + coverage_NS_std_TS2,  alpha=alpha)
    
    ax.set_title(synthetic_name[i], fontsize=text_size)  # Add title
    ax.grid(True)  # Add grid
    
    if i==0 or i==2:
        ax.set_ylabel('Reachability', fontsize=text_size)
    if i==2 or i==3:
        ax.set_xlabel('Number of evaluations', fontsize=text_size)
    if i == 0:  # Top-left subplot
       
        ax.legend(loc='lower right', fontsize=text_size)  # Add the legend with custom position and size   
        
plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05, wspace=0.2, hspace=0.2)       


# save_path = "/home/tang.1856/Jonathan/Novelty Search/Continuous_Synthetic_Code/Results/Noise/4DAckley/noise.mat"
# # Create a dictionary to store all data
# data_dict = {}

# # Iterate through datasets and save data
# for i in range(len(noise)):
#     data_dict[f'cost_NS_TS1_{i}'] = torch.load('/home/tang.1856/Jonathan/Novelty Search/Continuous_Synthetic_Code/Results/Noise/4DAckley/4DNoisyAckley_cost_list_NS_TS_considernoise_'+str(noise[i])+'.pt').numpy()
#     data_dict[f'coverage_NS_TS1_{i}'] = torch.load('/home/tang.1856/Jonathan/Novelty Search/Continuous_Synthetic_Code/Results/Noise/4DAckley/4DNoisyAckley_coverage_list_NS_TS_considernoise_'+str(noise[i])+'.pt').numpy()
#     data_dict[f'cost_NS_TS2_{i}'] = torch.load('/home/tang.1856/Jonathan/Novelty Search/Continuous_Synthetic_Code/Results/Noise/4DAckley/4DNoisyAckley_cost_list_NS_TS_noise_'+str(noise[i])+'.pt').numpy()
#     data_dict[f'coverage_NS_TS2_{i}'] = torch.load('/home/tang.1856/Jonathan/Novelty Search/Continuous_Synthetic_Code/Results/Noise/4DAckley/4DNoisyAckley_coverage_list_NS_TS_noise_'+str(noise[i])+'.pt').numpy()
    

# # Save the dictionary to a .mat file
# savemat(save_path, data_dict)



