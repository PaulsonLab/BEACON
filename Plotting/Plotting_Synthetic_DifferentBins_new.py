#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 16 14:38:55 2024

@author: tang.1856
"""
import torch
import matplotlib.pyplot as plt
from scipy.io import loadmat

synthetic = ['4DAckley', '4DAckley', '4DAckley', '4DRosen', '4DRosen', '4DRosen']
synthetic_name = ['Ackley 4D (grid=10)', 'Ackley 4D (grid=50)', 'Ackley 4D (grid=100)', 'Rosenbrock 4D (grid=10)', 'Rosenbrock 4D (grid=50)', 'Rosenbrock 4D (grid=100)']
bins = [10,50,100,10,50,100]
# synthetic = ['4DAckley', '4DAckley']
# synthetic_name = ['Ackley 4D (grid=10)', 'Ackley 4D (grid=50)']
# bins = [10,50]
save_path = "/home/tang.1856/Jonathan/Novelty Search/Continuous_Synthetic_Code/Results/Different Bins/different_grid.mat" # change the path
# path = '/home/tang.1856/Jonathan/Novelty Search/Continuous_Synthetic_Code/Results/Different Bins/'
loaded_data = loadmat(save_path)

fig, axes = plt.subplots(2, 3, figsize=(14, 14))  

for i, ax in enumerate(axes.flat):
    
    # cost_NS_TS1 = torch.load(path+synthetic[i]+'/'+synthetic[i]+'_TS_1_cost_list_NS_'+'bins'+str(bins[i])+'.pt')
    # coverage_NS_TS1 = torch.load(path+synthetic[i]+'/'+synthetic[i]+'_TS_1_coverage_list_NS_'+'bins'+str(bins[i])+'.pt')
    
    # cost_BO = torch.load(path+synthetic[i]+'/'+synthetic[i]+'_cost_list_MaxVar_'+'bins'+str(bins[i])+'.pt')
    # coverage_BO = torch.load(path+synthetic[i]+'/'+synthetic[i]+'_coverage_list_MaxVar_'+'bins'+str(bins[i])+'.pt')
    
    # cost_RS = torch.load(path+synthetic[i]+'/'+synthetic[i]+'_cost_list_RS_'+'bins'+str(bins[i])+'.pt')
    # coverage_RS = torch.load(path+synthetic[i]+'/'+synthetic[i]+'_coverage_list_RS_'+'bins'+str(bins[i])+'.pt')
    
    
    # cost_GA_NS_novel = torch.load(path+synthetic[i]+'/'+synthetic[i]+'_cost_list_GA_'+'bins'+str(bins[i])+'.pt')
    # coverage_GA_NS_novel = torch.load(path+synthetic[i]+'/'+synthetic[i]+'_coverage_list_GA_'+'bins'+str(bins[i])+'.pt')
    
    # cost_DEA = torch.load(path+synthetic[i]+'/'+synthetic[i]+'_cost_list_DEA_'+'bins'+str(bins[i])+'.pt')
    # coverage_DEA = torch.load(path+synthetic[i]+'/'+synthetic[i]+'_coverage_list_DEA_'+'bins'+str(bins[i])+'.pt')
    
    # cost_sobol = torch.load(path+synthetic[i]+'/'+synthetic[i]+'_cost_list_sobol_'+'bins'+str(bins[i])+'.pt')
    # coverage_sobol = torch.load(path+synthetic[i]+'/'+synthetic[i]+'_coverage_list_sobol_'+'bins'+str(bins[i])+'.pt')
    
    # cost_NS_xspace = torch.load(path+synthetic[i]+'/'+synthetic[i]+'_cost_list_NS_x_space_'+'bins'+str(bins[i])+'.pt')
    # coverage_NS_xspace = torch.load(path+synthetic[i]+'/'+synthetic[i]+'_coverage_list_NS_x_space_'+'bins'+str(bins[i])+'.pt')
    
    cost_NS_TS1 = torch.tensor(loaded_data['cost_NS_TS1_'+str(i)])
    coverage_NS_TS1 = torch.tensor(loaded_data['coverage_NS_TS1_'+str(i)])
    
    cost_BO = torch.tensor(loaded_data['cost_BO_'+str(i)])
    coverage_BO = torch.tensor(loaded_data['coverage_BO_'+str(i)])
    
    cost_RS = torch.tensor(loaded_data['cost_RS_'+str(i)])
    coverage_RS = torch.tensor(loaded_data['coverage_RS_'+str(i)])
    
    
    cost_GA_NS_novel = torch.tensor(loaded_data['cost_GA_NS_novel_'+str(i)])
    coverage_GA_NS_novel = torch.tensor(loaded_data['coverage_GA_NS_novel_'+str(i)])
    
    cost_DEA = torch.tensor(loaded_data['cost_DEA_'+str(i)])
    coverage_DEA = torch.tensor(loaded_data['coverage_DEA_'+str(i)])
    
    cost_sobol = torch.tensor(loaded_data['cost_sobol_'+str(i)])
    coverage_sobol = torch.tensor(loaded_data['coverage_sobol_'+str(i)])
    
    cost_NS_xspace = torch.tensor(loaded_data['cost_NS_xspace_'+str(i)])
    coverage_NS_xspace = torch.tensor(loaded_data['coverage_NS_xspace_'+str(i)])
    
    
    coverage_NS_mean_TS1 = torch.mean(coverage_NS_TS1, dim = 0)
    coverage_NS_std_TS1 = torch.std(coverage_NS_TS1, dim = 0)
    cost_NS_mean_TS1 = torch.mean(cost_NS_TS1, dim = 0)
    
    
    coverage_BO_mean = torch.mean(coverage_BO, dim = 0)
    coverage_BO_std = torch.std(coverage_BO, dim = 0)
    cost_BO_mean = torch.mean(cost_BO, dim = 0)
    
    coverage_RS_mean = torch.mean(coverage_RS, dim = 0)
    coverage_RS_std = torch.std(coverage_RS, dim = 0)
    cost_RS_mean = torch.mean(cost_RS, dim = 0)
    
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
    
    marker_interval = 10
    text_size = 14
    marker_size = 10
    weight='bold'
    linewidth = 4
    alpha = 0.3
   
    
    ax.plot(cost_NS_mean_TS1[::marker_interval], coverage_NS_mean_TS1[::marker_interval], label='BEACON', marker='X', markersize=marker_size, linewidth=linewidth)
    ax.plot(cost_BO_mean[::marker_interval], coverage_BO_mean[::marker_interval], label='MaxVar', marker='^', markersize=marker_size, linewidth=linewidth)
    ax.plot(cost_GA_NS_novel_mean, coverage_GA_NS_novel_mean, label='NS-EA', marker='>', markersize=marker_size, linewidth=linewidth )
    ax.plot(cost_DEA_mean, coverage_DEA_mean, label='NS-DEA', marker='D', markersize=marker_size, linewidth=linewidth )
    # ax.plot(cost_NS_xspace_mean[::marker_interval], coverage_NS_xspace_mean[::marker_interval], label='NS-EA-FS', marker='p', markersize=marker_size, linewidth=linewidth )
    ax.plot(cost_sobol_mean[::marker_interval], coverage_sobol_mean[::marker_interval], label='Sobol', marker='o', markersize=marker_size, linewidth=linewidth )
    ax.plot(cost_RS_mean[::marker_interval], coverage_RS_mean[::marker_interval], label='RS', marker='v', markersize=marker_size, linewidth=linewidth )
    
    ax.fill_between(cost_NS_mean_TS1, coverage_NS_mean_TS1 - coverage_NS_std_TS1, coverage_NS_mean_TS1 + coverage_NS_std_TS1,  alpha=alpha)
    ax.fill_between(cost_BO_mean, coverage_BO_mean - coverage_BO_std, coverage_BO_mean + coverage_BO_std,  alpha=alpha)
    ax.fill_between(cost_GA_NS_novel_mean, coverage_GA_NS_novel_mean - coverage_GA_NS_novel_std, coverage_GA_NS_novel_mean + coverage_GA_NS_novel_std,  alpha=alpha)
    ax.fill_between(cost_DEA_mean, coverage_DEA_mean - coverage_DEA_std, coverage_DEA_mean + coverage_DEA_std,  alpha=alpha)
    # ax.fill_between(cost_NS_xspace_mean, coverage_NS_xspace_mean - coverage_NS_xspace_std, coverage_NS_xspace_mean + coverage_NS_xspace_std,  alpha=alpha)
    ax.fill_between(cost_sobol_mean, coverage_sobol_mean - coverage_sobol_std, coverage_sobol_mean + coverage_sobol_std,  alpha=alpha)
    ax.fill_between(cost_RS_mean, coverage_RS_mean - coverage_RS_std, coverage_RS_mean + coverage_RS_std,  alpha=alpha)
    
    ax.set_ylim(0.05,1.05)
    
    ax.set_title(synthetic_name[i], fontsize=text_size )  # Add title
    ax.grid(True)  # Add grid
    
    
    # Add a legend to the top-left subplot (subplot 0)
    if i == 0:  # Top-left subplot
        # ax.set_ylabel('Reachability', fontsize=14)
        ax.legend(loc='upper left', fontsize=text_size )  # Add the legend with custom position and size
    if i==0 or i==3: 
        ax.set_ylabel('Reachability', fontsize=text_size )
    if i==3 or i==4 or i==5:
        ax.set_xlabel('Number of evaluations', fontsize=text_size )
        

plt.subplots_adjust(left=0.1, right=0.95, top=0.95, bottom=0.08, wspace=0.2, hspace=0.2)

# from scipy.io import savemat
# save_path = "/home/tang.1856/Jonathan/Novelty Search/Continuous_Synthetic_Code/Results/Different Bins/different_grid.mat"
# # Create a dictionary to store all data
# data_dict = {}

# # Iterate through datasets and save data
# for i in range(len(synthetic)):
#     data_dict[f'cost_NS_TS1_{i}'] = torch.load(path + synthetic[i] + '/' + synthetic[i] + '_TS_1_cost_list_NS_' + 'bins' + str(bins[i]) + '.pt').numpy()
#     data_dict[f'coverage_NS_TS1_{i}'] = torch.load(path + synthetic[i] + '/' + synthetic[i] + '_TS_1_coverage_list_NS_' + 'bins' + str(bins[i]) + '.pt').numpy()
#     data_dict[f'cost_BO_{i}'] = torch.load(path + synthetic[i] + '/' + synthetic[i] + '_cost_list_MaxVar_' + 'bins' + str(bins[i]) + '.pt').numpy()
#     data_dict[f'coverage_BO_{i}'] = torch.load(path + synthetic[i] + '/' + synthetic[i] + '_coverage_list_MaxVar_' + 'bins' + str(bins[i]) + '.pt').numpy()
#     data_dict[f'cost_RS_{i}'] = torch.load(path + synthetic[i] + '/' + synthetic[i] + '_cost_list_RS_' + 'bins' + str(bins[i]) + '.pt').numpy()
#     data_dict[f'coverage_RS_{i}'] = torch.load(path + synthetic[i] + '/' + synthetic[i] + '_coverage_list_RS_' + 'bins' + str(bins[i]) + '.pt').numpy()
#     data_dict[f'cost_GA_NS_novel_{i}'] = torch.load(path + synthetic[i] + '/' + synthetic[i] + '_cost_list_GA_' + 'bins' + str(bins[i]) + '.pt').numpy()
#     data_dict[f'coverage_GA_NS_novel_{i}'] = torch.load(path + synthetic[i] + '/' + synthetic[i] + '_coverage_list_GA_' + 'bins' + str(bins[i]) + '.pt').numpy()
#     data_dict[f'cost_DEA_{i}'] = torch.load(path + synthetic[i] + '/' + synthetic[i] + '_cost_list_DEA_' + 'bins' + str(bins[i]) + '.pt').numpy()
#     data_dict[f'coverage_DEA_{i}'] = torch.load(path + synthetic[i] + '/' + synthetic[i] + '_coverage_list_DEA_' + 'bins' + str(bins[i]) + '.pt').numpy()
#     data_dict[f'cost_sobol_{i}'] = torch.load(path + synthetic[i] + '/' + synthetic[i] + '_cost_list_sobol_' + 'bins' + str(bins[i]) + '.pt').numpy()
#     data_dict[f'coverage_sobol_{i}'] = torch.load(path + synthetic[i] + '/' + synthetic[i] + '_coverage_list_sobol_' + 'bins' + str(bins[i]) + '.pt').numpy()
#     data_dict[f'cost_NS_xspace_{i}'] = torch.load(path + synthetic[i] + '/' + synthetic[i] + '_cost_list_NS_x_space_' + 'bins' + str(bins[i]) + '.pt').numpy()
#     data_dict[f'coverage_NS_xspace_{i}'] = torch.load(path + synthetic[i] + '/' + synthetic[i] + '_coverage_list_NS_x_space_' + 'bins' + str(bins[i]) + '.pt').numpy()

# # Save the dictionary to a .mat file
# savemat(save_path, data_dict)



# # Load the .mat file
# loaded_data = loadmat(save_path)

# # Access the data (keys are the variable names in the dictionary)
# cost_NS_TS1_0 = loaded_data['cost_NS_TS1_0']
# coverage_NS_TS1_0 = loaded_data['coverage_NS_TS1_0']














