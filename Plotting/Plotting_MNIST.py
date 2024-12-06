#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  1 13:42:10 2024

@author: tang.1856
"""
import torch
import numpy as np
import matplotlib.pyplot as plt


text_size = 34
marker_size = 18
linewidth=4
marker_interval = 8
weight='bold'
alpha = 0.3

indice = [0,3,5,7,9,10,11,12,13,15]

reachability_list_BEACON = np.array(torch.load('/home/tang.1856/BEACON/BEACON/MNIST/Results_MNIST/MNIST_reachability_BEACON.pt'))[indice]
mean = np.mean(reachability_list_BEACON, axis=0)
std = np.std(reachability_list_BEACON, axis=0)

# reachability_list_RS = np.array(torch.load('Results/MNIST_reachability_RS_40.pt'))
# meanRS = np.mean(reachability_list_RS, axis=0)
# stdRS = np.std(reachability_list_RS, axis=0)

# reachability_list_FS = np.array(torch.load('/home/tang.1856/BEACON/Results/MNIST_coverage_list_NS_xspace_40.pt')) #2
# meanFS = np.mean(reachability_list_FS, axis=0)
# stdFS = np.std(reachability_list_FS, axis=0)

# reachability_list_GA = np.array(torch.load('/home/tang.1856/BEACON/Results/MNIST_coverage_list_NSEA.pt')) #2
# meanGA = np.mean(reachability_list_GA, axis=0)
# stdGA = np.std(reachability_list_GA, axis=0)

reachability_list_BEACON_bc = np.array(torch.load('/home/tang.1856/BEACON/BEACON/MNIST/Results_MNIST/MNIST_reachability_BEACON_bc.pt'))[indice]
mean_bc = np.median(reachability_list_BEACON_bc, axis=0)
std_bc = np.std(reachability_list_BEACON_bc, axis=0)

reachability_list_RS = np.array(torch.load('/home/tang.1856/BEACON/BEACON/MNIST/Results_MNIST/MNIST_reachability_RS.pt'))[indice]
meanRS = np.median(reachability_list_RS, axis=0)
stdRS = np.std(reachability_list_RS, axis=0)

# reachability_list_FS = np.array(torch.load('/home/tang.1856/BEACON/BEACON/MNIST/Results_MNIST/MNIST_coverage_list_NS_FS.pt'))[indice]
# meanFS = np.median(reachability_list_FS, axis=0)
# stdFS = np.std(reachability_list_FS, axis=0)

reachability_list_GA = np.array(torch.load('/home/tang.1856/BEACON/BEACON/MNIST/Results_MNIST/MNIST_coverage_list_NSEA.pt'))[indice]
meanGA = np.median(reachability_list_GA, axis=0)
stdGA = np.std(reachability_list_GA, axis=0)

# Create an x-axis for plotting
x = np.arange(mean.shape[0])
xGA = np.arange(meanGA.shape[0])*10
# Plot the mean and the standard deviation
# plt.figure(figsize=(10, 5))
# plt.plot(x, mean, label='BEACON')
# plt.fill_between(x, mean - std, mean + std,  alpha=0.2)
# plt.plot(x, meanRS, label='RS')
# plt.fill_between(x, meanRS - stdRS, meanRS + stdRS,  alpha=0.2)
# plt.plot(x, meanFS, label='NS-FS')
# plt.fill_between(x, meanFS - stdFS, meanFS + stdFS,  alpha=0.2)
# plt.xlabel('Iteration')
# plt.ylabel('Reachability')
# plt.title('Mean of MNIST')
# plt.legend()
# plt.show()

plt.figure(figsize=(16,14))
mean = np.median(reachability_list_BEACON, axis=0)
std = np.std(reachability_list_BEACON, axis=0)
max_ = np.max(reachability_list_BEACON, axis=0)
min_ = np.min(reachability_list_BEACON, axis=0)

meanRS = np.median(reachability_list_RS, axis=0)
stdRS = np.std(reachability_list_RS, axis=0)
max_RS = np.max(reachability_list_RS, axis=0)
min_RS = np.min(reachability_list_RS, axis=0)

# meanFS = np.median(reachability_list_FS, axis=0)
# stdFS = np.std(reachability_list_FS, axis=0)
# max_FS = np.max(reachability_list_FS, axis=0)
# min_FS = np.min(reachability_list_FS, axis=0)

meanGA = np.median(reachability_list_GA, axis=0)
stdGA = np.std(reachability_list_GA, axis=0)
max_GA = np.max(reachability_list_GA, axis=0)
min_GA = np.min(reachability_list_GA, axis=0)

# Create an x-axis for plotting
x = np.arange(mean.shape[0])

# Plot the mean and the standard deviation
# plt.figure(figsize=(10, 5))
plt.plot(x, mean, label='BEACON', marker='X', markersize=marker_size, linewidth=linewidth,color='#1f77b4')
plt.plot(x, mean_bc, label='BEACON-bc', marker='X', markersize=marker_size, linewidth=linewidth,color='orange')
# plt.fill_between(x, min_, max_,  alpha=0.2)

# plt.fill_between(x, min_RS, max_RS,  alpha=0.2)

# plt.fill_between(x, min_FS, max_FS,  alpha=0.2)
plt.plot(xGA, meanGA, label='NS-EA', marker='>', markersize=marker_size, linewidth=linewidth,color='#2ca02c')
# plt.plot(x, meanFS, label='NS-FS', marker='p', markersize=marker_size, linewidth=linewidth,color='#9467bd')
plt.plot(x, meanRS, label='RS', marker='v', markersize=marker_size, linewidth=linewidth,color='#e377c2')
plt.xlabel('Iteration', fontsize=text_size, fontweight=weight)
plt.ylabel('Number of discovered behaviors', fontsize=text_size, fontweight=weight)
# plt.title('MNIST novelty search in pixel space', fontsize=text_size, fontweight=weight)
# plt.title('Mean and ±1 Standard Deviation')
plt.legend(prop={'weight':'bold','size':text_size-4})
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

import pandas as pd
import seaborn as sns
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

    'NS-FS': '#9467bd',

    'RS': '#e377c2',

}
# Violin plot
text_size = 16
plt.figure(figsize=(14,12))

# Convert tensors to lists and create a DataFrame
data = {name: tensor.tolist() for name, tensor in tensors.items()}
df = pd.DataFrame(data)

# Melt the DataFrame to long format
df_melted = df.melt(var_name='Tensor', value_name='Value')

# Create the violin plot with the same colors
sns.violinplot(x='Tensor', y='Value', data=df_melted, palette=palette, width=0.4, inner='point',cut=0)
sns.stripplot(x='Tensor', y='Value', data=df_melted, jitter=True, size=4, color='k', alpha=0.6)
plt.ylabel('Number of final behaviors', fontsize=text_size, fontweight='bold')
plt.xlabel('')
plt.xticks(rotation=45, fontsize=12, fontweight='bold')
plt.yticks(fontsize=12, fontweight='bold')
plt.grid(alpha=0.5, linewidth=1.0)

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

plt.show()


from scipy.io import savemat
save_path = "/home/tang.1856/BEACON/BEACON/Plotting/MNIST.mat"
# Create a dictionary to store all data
data_dict = {}

# Iterate through datasets and save data
for i in range(1):
    data_dict[f'coverage_BEACON_{i}'] = np.array(torch.load('/home/tang.1856/BEACON/BEACON/MNIST/Results_MNIST/MNIST_reachability_BEACON.pt'))[indice]
    data_dict[f'coverage_BEACON_bc_{i}'] = np.array(torch.load('/home/tang.1856/BEACON/BEACON/MNIST/Results_MNIST/MNIST_reachability_BEACON_bc.pt'))[indice]
    data_dict[f'cost_RS_{i}'] = np.array(torch.load('/home/tang.1856/BEACON/BEACON/MNIST/Results_MNIST/MNIST_reachability_RS.pt'))[indice]
    data_dict[f'coverage_NSEA_{i}'] = np.array(torch.load('/home/tang.1856/BEACON/BEACON/MNIST/Results_MNIST/MNIST_coverage_list_NSEA.pt'))[indice]
   
# Save the dictionary to a .mat file
savemat(save_path, data_dict)