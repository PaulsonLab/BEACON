#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  6 16:43:21 2024

@author: tang.1856
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
from scipy.io import loadmat

# Define your synthetic function and generate data
def synthetic_function(x):
    y1 = np.sin(x[0]) * np.cos(x[1]) + x[2] * np.exp(-x[0]**2) * np.cos(x[0] + x[1]) + 0.01*np.sin((x[3] + x[4] + x[5]))
    y2 = np.sin(x[3]) * np.cos(x[4]) + x[5] * np.exp(-x[3]**2) * np.sin(x[3] + x[4]) + 0.01*np.cos((x[0] + x[1] + x[2]))
    return np.array([y1, y2])



# Create a large figure with 2 subplots
fig = plt.figure(figsize=(12, 12))
outer_gs = fig.add_gridspec(2, 1, wspace=0.3)  # Two subplots side by side

text_size = 16

inputs = np.random.uniform(-5, 5, (20000, 6))  # Uniformly distributed in a range
outcomes = np.array([synthetic_function(x) for x in inputs])
x = outcomes[:, 0]
y = outcomes[:, 1]

# Add the first subplot for the scatter plot with marginal histograms
gs = outer_gs[0].subgridspec(2, 2, width_ratios=(4, 1), height_ratios=(1, 4), wspace=0.05, hspace=0.05)

# Create axes for the first subplot
ax = fig.add_subplot(gs[1, 0])  # Main scatter plot
ax_histx = fig.add_subplot(gs[0, 0], sharex=ax)  # Marginal histogram for x-axis
ax_histy = fig.add_subplot(gs[1, 1], sharey=ax)  # Marginal histogram for y-axis

# Scatter plot and histograms
ax.scatter(x, y, alpha=0.5, edgecolors='none', s=4)
ax.grid(True)
ax.set_xticks(np.linspace(-5, 5, 11))
ax.set_yticks(np.linspace(-5, 5, 11))
ax.set_xlim(-5, 5)
ax.set_ylim(-5, 5)

binwidth = 0.25
bins = np.arange(-5, 5 + binwidth, binwidth)
ax_histx.hist(x, bins=bins, color='cornflowerblue', edgecolor='black', alpha=0.5)
ax_histy.hist(y, bins=bins, orientation='horizontal', color='cornflowerblue', edgecolor='black', alpha=0.5)

# Hide tick labels for the marginal axes
ax_histx.tick_params(axis="x", labelbottom=False)
ax_histy.tick_params(axis="y", labelleft=False)

# Customize labels
ax.set_xlabel('y1', fontsize=text_size)
ax.set_ylabel('y2', fontsize=text_size)

# second figure
synthetic = 'Cluster'
save_path = "/home/tang.1856/BEACON/BEACON/Plotting/multioutcome.mat"
loaded_data = loadmat(save_path)
i=0

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




marker_size = 18
linewidth=4
marker_interval = 20
weight='bold'
alpha = 0.3

ax2 = fig.add_subplot(outer_gs[1])

ax2.plot(cost_NS_mean_TS1[::marker_interval], coverage_NS_mean_TS1[::marker_interval], label='BEACON', marker='X', markersize=marker_size, linewidth=linewidth)
ax2.plot(cost_BO_mean[::marker_interval], coverage_BO_mean[::marker_interval], label='MaxVar', marker='^', markersize=marker_size, linewidth=linewidth)
ax2.plot(cost_GA_NS_novel_mean, coverage_GA_NS_novel_mean, label='NS-EA', marker='>', markersize=marker_size, linewidth=linewidth )
# plt.plot(cost_DEA_mean, coverage_DEA_mean, label='NS-DEA', marker='D', markersize=marker_size, linewidth=linewidth )
# plt.plot(cost_NS_xspace_mean[::marker_interval], coverage_NS_xspace_mean[::marker_interval], label='NS-FS', marker='p', markersize=marker_size, linewidth=linewidth )
ax2.plot(cost_sobol_mean[::marker_interval], coverage_sobol_mean[::marker_interval], label='Sobol', marker='o', markersize=marker_size, linewidth=linewidth )
ax2.plot(cost_RS_mean[::marker_interval], coverage_RS_mean[::marker_interval], label='RS', marker='v', markersize=marker_size, linewidth=linewidth )
# plt.plot(cost_EI_mean[::marker_interval], coverage_EI_mean[::marker_interval], label='EI', marker='*', markersize=marker_size, linewidth=linewidth )

ax2.fill_between(cost_NS_mean_TS1, coverage_NS_mean_TS1 - coverage_NS_std_TS1, coverage_NS_mean_TS1 + coverage_NS_std_TS1,  alpha=alpha)
ax2.fill_between(cost_BO_mean, coverage_BO_mean - coverage_BO_std, coverage_BO_mean + coverage_BO_std,  alpha=alpha)
ax2.fill_between(cost_GA_NS_novel_mean, coverage_GA_NS_novel_mean - coverage_GA_NS_novel_std, coverage_GA_NS_novel_mean + coverage_GA_NS_novel_std,  alpha=alpha)
# plt.fill_between(cost_DEA_mean, coverage_DEA_mean - coverage_DEA_std, coverage_DEA_mean + coverage_DEA_std,  alpha=alpha)
# plt.fill_between(cost_NS_xspace_mean, coverage_NS_xspace_mean - coverage_NS_xspace_std, coverage_NS_xspace_mean + coverage_NS_xspace_std,  alpha=alpha)
ax2.fill_between(cost_sobol_mean, coverage_sobol_mean - coverage_sobol_std, coverage_sobol_mean + coverage_sobol_std,  alpha=alpha)
ax2.fill_between(cost_RS_mean, coverage_RS_mean - coverage_RS_std, coverage_RS_mean + coverage_RS_std,  alpha=alpha)
# Add the second subplot (e.g., another figure/plot)

ax2.set_xlabel('Number of evaluations', fontsize=text_size)
ax2.set_ylabel('Reachability', fontsize=text_size)
ax2.legend(prop={'size':text_size})
ax2.grid(alpha=0.5, linewidth=2.0)

        


