#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  1 13:42:10 2024

@author: tang.1856
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy.io import loadmat

text_size = 16
marker_size = 14
linewidth=4
marker_interval = 8
weight='bold'
alpha = 0.3

indice = [0,3,5,7,9,10,11,12,13,15]

save_path = "/home/tang.1856/BEACON/BEACON/Plotting/MNIST.mat"
loaded_data = loadmat(save_path)

i=0
reachability_list_BEACON_bc = (loaded_data['coverage_BEACON_bc_'+str(i)])

reachability_list_BEACON = (loaded_data['coverage_BEACON_'+str(i)])

reachability_list_RS = (loaded_data['cost_RS_'+str(i)])

reachability_list_GA = (loaded_data['coverage_NSEA_'+str(i)])


fig, axes = plt.subplots(1, 2, figsize=(16,12))  # 3 rows, 3 columns
# Add plots and legends to subplots
for i, ax in enumerate(axes.flat):
    
    # reachability_list_BEACON = np.array(torch.load('/home/tang.1856/BEACON/BEACON/MNIST/Results_MNIST/MNIST_reachability_BEACON.pt'))[indice]
    mean = np.mean(reachability_list_BEACON, axis=0)
    std = np.std(reachability_list_BEACON, axis=0)
    
    
    # reachability_list_BEACON_bc = np.array(torch.load('/home/tang.1856/BEACON/BEACON/MNIST/Results_MNIST/MNIST_reachability_BEACON_bc.pt'))[indice]
    mean_bc = np.median(reachability_list_BEACON_bc, axis=0)
    std_bc = np.std(reachability_list_BEACON_bc, axis=0)
    
    # reachability_list_RS = np.array(torch.load('/home/tang.1856/BEACON/BEACON/MNIST/Results_MNIST/MNIST_reachability_RS.pt'))[indice]
    meanRS = np.median(reachability_list_RS, axis=0)
    stdRS = np.std(reachability_list_RS, axis=0)
    
    # reachability_list_GA = np.array(torch.load('/home/tang.1856/BEACON/BEACON/MNIST/Results_MNIST/MNIST_coverage_list_NSEA.pt'))[indice]
    meanGA = np.median(reachability_list_GA, axis=0)
    stdGA = np.std(reachability_list_GA, axis=0)
    
    # Create an x-axis for plotting
    x = np.arange(mean.shape[0])
    xGA = np.arange(meanGA.shape[0])*10
    
    
    # plt.figure(figsize=(16,14))
    mean = np.median(reachability_list_BEACON, axis=0)
    std = np.std(reachability_list_BEACON, axis=0)
    max_ = np.max(reachability_list_BEACON, axis=0)
    min_ = np.min(reachability_list_BEACON, axis=0)
    
    meanRS = np.median(reachability_list_RS, axis=0)
    stdRS = np.std(reachability_list_RS, axis=0)
    max_RS = np.max(reachability_list_RS, axis=0)
    min_RS = np.min(reachability_list_RS, axis=0)
    
    
    meanGA = np.median(reachability_list_GA, axis=0)
    stdGA = np.std(reachability_list_GA, axis=0)
    max_GA = np.max(reachability_list_GA, axis=0)
    min_GA = np.min(reachability_list_GA, axis=0)
    
    # Create an x-axis for plotting
    x = np.arange(mean.shape[0])
    
    tensors = {
        'BEACON':reachability_list_BEACON[:,-1], 
        'BEACON_bc':reachability_list_BEACON_bc[:,-1], 
        'NS-EA':reachability_list_GA[:,-1],    
        # 'NS-FS':reachability_list_FS[:,-1],   
        'RS':reachability_list_RS[:,-1],
          
    }
    palette = {
        'BEACON': '#1f77b4',
        'BEACON_bc': 'orange',
        'NS-EA': '#2ca02c',
    
        # 'NS-FS': '#9467bd',
    
        'RS': '#e377c2',
    
    }
   
    
    # plt.figure(figsize=(14,12))
    
    if i==0:
        # Plot the mean and the standard deviation
        # plt.figure(figsize=(10, 5))
        ax.plot(x, mean, label='BEACON', marker='o', markersize=marker_size, linewidth=linewidth,color='#1f77b4')
        ax.plot(x, mean_bc, label='BEACON-bc', marker='X', markersize=marker_size, linewidth=linewidth,color='orange')
        # plt.fill_between(x, min_, max_,  alpha=0.2)
        
        # plt.fill_between(x, min_RS, max_RS,  alpha=0.2)
        
        # plt.fill_between(x, min_FS, max_FS,  alpha=0.2)
        ax.plot(xGA, meanGA, label='NS-EA', marker='>', markersize=marker_size, linewidth=linewidth,color='#2ca02c')
        # plt.plot(x, meanFS, label='NS-FS', marker='p', markersize=marker_size, linewidth=linewidth,color='#9467bd')
        ax.plot(x, meanRS, label='RS', marker='v', markersize=marker_size, linewidth=linewidth,color='#e377c2')
        ax.set_xlabel('Iteration', fontsize=text_size)
        ax.set_ylabel('Number of discovered behaviors', fontsize=text_size)
        # plt.title('MNIST novelty search in pixel space', fontsize=text_size, fontweight=weight)
        # plt.title('Mean and ±1 Standard Deviation')
        ax.legend(prop={'size':text_size})
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
        
        ax.grid(alpha=0.5, linewidth=2.0)
    
    
    
    
    else:
        # Convert tensors to lists and create a DataFrame
        data = {name: tensor.tolist() for name, tensor in tensors.items()}
        df = pd.DataFrame(data)
        
        # Melt the DataFrame to long format
        df_melted = df.melt(var_name='Tensor', value_name='Value')
        
        # Create the violin plot with the same colors
        sns.violinplot(x='Tensor', y='Value', data=df_melted, palette=palette, width=0.4, inner='point',cut=0)
        sns.stripplot(x='Tensor', y='Value', data=df_melted, jitter=True, size=4, color='k', alpha=0.6)
        ax.set_ylabel('Number of final behaviors', fontsize=text_size)
        ax.set_xlabel('')
        # ax.set_xticks(rotation=45, fontsize=12, fontweight='bold')
        # ax.set_yticks(fontsize=12, fontweight='bold')
        ax.grid(alpha=0.5, linewidth=2.0)
        
        # ax = plt.gca()
        for label in ax.get_xticklabels():
            label.set_fontsize(text_size)
        #     label.set_fontweight('bold')
        
        # for label in ax.get_yticklabels():
        #     label.set_fontsize(text_size)
        #     label.set_fontweight('bold')
        # ax.spines['top'].set_linewidth(2)
        # ax.spines['bottom'].set_linewidth(2)
        # ax.spines['left'].set_linewidth(2)
        # ax.spines['right'].set_linewidth(2)
        
        # plt.show()

plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.08, wspace=0.2, hspace=0.1)    
# from scipy.io import savemat
# save_path = "/home/tang.1856/BEACON/BEACON/Plotting/MNIST.mat"
# # Create a dictionary to store all data
# data_dict = {}

# # Iterate through datasets and save data
# for i in range(1):
#     data_dict[f'coverage_BEACON_{i}'] = np.array(torch.load('/home/tang.1856/BEACON/BEACON/MNIST/Results_MNIST/MNIST_reachability_BEACON.pt'))[indice]
#     data_dict[f'coverage_BEACON_bc_{i}'] = np.array(torch.load('/home/tang.1856/BEACON/BEACON/MNIST/Results_MNIST/MNIST_reachability_BEACON_bc.pt'))[indice]
#     data_dict[f'cost_RS_{i}'] = np.array(torch.load('/home/tang.1856/BEACON/BEACON/MNIST/Results_MNIST/MNIST_reachability_RS.pt'))[indice]
#     data_dict[f'coverage_NSEA_{i}'] = np.array(torch.load('/home/tang.1856/BEACON/BEACON/MNIST/Results_MNIST/MNIST_coverage_list_NSEA.pt'))[indice]
   
# # Save the dictionary to a .mat file
# savemat(save_path, data_dict)