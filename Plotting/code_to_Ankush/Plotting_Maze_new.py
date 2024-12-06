# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# """
# Created on Sat Mar 16 14:38:55 2024

# @author: tang.1856
# """


import torch
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
# import numpy as np
from scipy.io import loadmat

plt.rcParams['font.size'] = 8
# Load your tensors (replace with actual data loading)
synthetic = 'Maze'

save_path = "Maze.mat"
loaded_data = loadmat(save_path)

i=0
cost_NS_TS1 = torch.tensor(loaded_data['cost_NS_TS1_'+str(i)])
coverage_NS_TS1 =torch.tensor(loaded_data['coverage_NS_TS1_'+str(i)])

cost_BO = torch.tensor(loaded_data['cost_BO_'+str(i)])
coverage_BO = torch.tensor(loaded_data['coverage_BO_'+str(i)])

cost_MaxVar = torch.tensor(loaded_data['cost_MaxVar_'+str(i)])
coverage_MaxVar = torch.tensor(loaded_data['coverage_MaxVar_'+str(i)])

cost_RS = torch.tensor(loaded_data['cost_RS_'+str(i)])
coverage_RS = torch.tensor(loaded_data['coverage_RS_'+str(i)])

cost_GA_NS_novel = torch.tensor(loaded_data['cost_GA_NS_novel_'+str(i)])
coverage_GA_NS_novel = torch.tensor(loaded_data['coverage_GA_NS_novel_'+str(i)])

cost_DEA = torch.tensor(loaded_data['cost_DEA_'+str(i)])
coverage_DEA = torch.tensor(loaded_data['coverage_DEA_'+str(i)])

cost_sobol = torch.tensor(loaded_data['cost_Sobol_'+str(i)])
coverage_sobol = torch.tensor(loaded_data['coverage_Sobol_'+str(i)])

fig, axes = plt.subplots(2, 1, figsize=(6, 6), dpi=150)  # 3 rows, 3 columns
# Add plots and legends to subplots
for i, ax in enumerate(axes.flat):
    
    
    
    # Compute means and stds for plotting
    coverage_NS_mean_TS1 = torch.mean(coverage_NS_TS1, dim=0)
    coverage_NS_std_TS1 = torch.std(coverage_NS_TS1, dim=0)
    cost_NS_mean_TS1 = torch.mean(cost_NS_TS1, dim=0)
    
    coverage_BO_mean = torch.mean(coverage_BO, dim=0)
    coverage_BO_std = torch.std(coverage_BO, dim=0)
    cost_BO_mean = torch.mean(cost_BO, dim=0)
    
    coverage_MaxVar_mean = torch.mean(coverage_MaxVar, dim=0)
    coverage_MaxVar_std = torch.std(coverage_MaxVar, dim=0)
    cost_MaxVar_mean = torch.mean(cost_MaxVar, dim=0)
    
    coverage_RS_mean = torch.mean(coverage_RS, dim=0)
    coverage_RS_std = torch.std(coverage_RS, dim=0)
    cost_RS_mean = torch.mean(cost_RS, dim=0)
    
    coverage_GA_NS_novel_mean = torch.mean(coverage_GA_NS_novel, dim=0)
    coverage_GA_NS_novel_std = torch.std(coverage_GA_NS_novel, dim=0)
    cost_GA_NS_novel_mean = torch.mean(cost_GA_NS_novel, dim=0)
    
    coverage_DEA_mean = torch.mean(coverage_DEA, dim=0)
    coverage_DEA_std = torch.std(coverage_DEA, dim=0)
    cost_DEA_mean = torch.mean(cost_DEA, dim=0)
    
    # coverage_NS_xspace_mean = torch.mean(coverage_NS_xspace, dim=0)
    # coverage_NS_xspace_std = torch.std(coverage_NS_xspace, dim=0)
    # cost_NS_xspace_mean = torch.mean(cost_NS_xspace, dim=0)
    
    coverage_sobol_mean = torch.mean(coverage_sobol, dim=0)
    coverage_sobol_std = torch.std(coverage_sobol, dim=0)
    cost_sobol_mean = torch.mean(cost_sobol, dim=0)
    
    marker_interval = 8
    text_size = 8
    marker_size = 4
    weight='normal'
    linewidth = 1
    alpha = 0.3
    
    # Define a consistent color palette
    palette = {
        'BEACON': '#1f77b4',
        'MaxVar': '#ff7f0e',
        'NS-EA': '#2ca02c',
        'NS-DEA': '#d62728',
        # 'NS-FS': '#9467bd',
        'Sobol': '#8c564b',
        'RS': '#e377c2',
        'EI': '#7f7f7f'
    }
    
    tensors = {
        'BEACON':coverage_NS_TS1[:,-1],
        'MaxVar':coverage_MaxVar[:,-1],
        'NS-EA':coverage_GA_NS_novel[:,-1],
        'NS-DEA':coverage_DEA[:,-1],
        # 'NS-FS':coverage_NS_xspace[:,-1],
        'Sobol':coverage_sobol[:,-1],
        'RS':coverage_RS[:,-1],
        'EI':coverage_BO[:,-1],       
    }
    
    
    # Line plot
    # plt.figure(figsize=(12, 10))
    if i==0:
        ax.plot(cost_NS_mean_TS1[::marker_interval], coverage_NS_mean_TS1[::marker_interval], label='BEACON', marker='o', markersize=marker_size, linewidth=linewidth, color=palette['BEACON'])
        ax.plot(cost_MaxVar_mean[::marker_interval], coverage_MaxVar_mean[::marker_interval], label='MaxVar', marker='^', markersize=marker_size, linewidth=linewidth, color=palette['MaxVar'])
        ax.plot(cost_GA_NS_novel_mean, coverage_GA_NS_novel_mean, label='NS-EA', marker='>', markersize=marker_size, linewidth=linewidth, color=palette['NS-EA'])
        ax.plot(cost_DEA_mean, coverage_DEA_mean, label='NS-DEA', marker='D', markersize=marker_size, linewidth=linewidth, color=palette['NS-DEA'])
        # plt.plot(cost_NS_xspace_mean[::marker_interval], coverage_NS_xspace_mean[::marker_interval], label='NS-FS', marker='p', markersize=marker_size, linewidth=linewidth, color=palette['NS-FS'])
        ax.plot(cost_sobol_mean[::marker_interval], coverage_sobol_mean[::marker_interval], label='Sobol', marker='+', markersize=marker_size, linewidth=linewidth, color=palette['Sobol'])
        ax.plot(cost_RS_mean[::marker_interval], coverage_RS_mean[::marker_interval], label='RS', marker='v', markersize=marker_size, linewidth=linewidth, color=palette['RS'])
        ax.plot(cost_BO_mean[::marker_interval], coverage_BO_mean[::marker_interval], label='EI', marker='<', markersize=marker_size, linewidth=linewidth, color=palette['EI'])
        
        ax.set_ylim(0.48, 1.02)
        ax.set_xlabel('Number of evaluations', fontsize='large')
        ax.set_ylabel('Best Reward', fontsize='large')
        ax.legend(prop={'weight':'normal','size':'medium'}, loc='best')
    
        # plt.tick_params(axis='both', which='both', width=2)
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
        
        ax.grid(alpha=0.5, linewidth=2)
        
    else:
        # Violin plot
        text_size = 18
        # plt.figure(figsize=(12,10))
        
        # Convert tensors to lists and create a DataFrame
        data = {name: tensor.tolist() for name, tensor in tensors.items()}
        df = pd.DataFrame(data)
        
        # Melt the DataFrame to long format
        df_melted = df.melt(var_name='Tensor', value_name='Value')
        
        # Create the violin plot with the same colors
        sns.violinplot(x='Tensor', y='Value', data=df_melted, palette=palette, width=0.4, inner='point', cut=0)
        
        ax.set_ylabel('Final Reward Value', fontsize='large')
        ax.set_xlabel('', fontsize='large')
        # ax.set_xticks(rotation=45, fontsize=12)
        # ax.set_yticks(fontsize=12, fontweight='bold')
        ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
        ax.grid(alpha=0.5, linewidth=1.0)            
            
plt.tight_layout()
plt.savefig('Maze.png', dpi=300)
plt.show()