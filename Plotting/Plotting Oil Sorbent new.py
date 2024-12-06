#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  5 16:38:49 2024

@author: tang.1856
"""

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from scipy.io import loadmat
import torch

save_path = "/home/tang.1856/BEACON/BEACON/Plotting/OilSorbent.mat"
loaded_data = loadmat(save_path)

i=0
cost_BEACON = torch.tensor(loaded_data['cost_BEACON_'+str(i)])
coverage_BEACON = torch.tensor(loaded_data['coverage_BEACON_'+str(i)])

cost_BEACON_constraint = torch.tensor(loaded_data['cost_BEACON_bc_'+str(i)])
coverage_BEACON_constraint = torch.tensor(loaded_data['coverage_BEACON_bc_'+str(i)])

cost_MaxVar_constraint = torch.tensor(loaded_data['cost_MaxVar_'+str(i)])
coverage_MaxVar_constraint = torch.tensor(loaded_data['coverage_MaxVar_'+str(i)])

cost_RS_constraint = torch.tensor(loaded_data['cost_RS_'+str(i)])
coverage_RS_constraint = torch.tensor(loaded_data['coverage_RS_'+str(i)])

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

# Sample data for the plot
# cost_NS_mean_TS1 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
# coverage_NS_mean_TS1 = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]
marker_interval = 10
marker_size = 10
linewidth = 2
text_size = 16

# Create a 1x2 subfigure layout
fig, axes = plt.subplots(2, 1, figsize=(6, 6), dpi=150)  # 1 row, 2 columns

# Second subplot: display the PNG image
img = mpimg.imread('/home/tang.1856/BEACON/BEACON/Plotting/OilSorbent_distribution.png')  # Replace 'your_image.png' with your file path
axes[0].imshow(img)
axes[0].axis('off')  # Hide axes for the image
# axes[0].set_title('PNG Image')

# First subplot: plot the line chart
axes[1].plot(cost_NS_mean_TS1[::marker_interval], coverage_NS_mean_TS1[::marker_interval], label='BEACON', marker='X', markersize=marker_size,linewidth=linewidth)
axes[1].plot(cost_BO_mean[::marker_interval], coverage_BO_mean[::marker_interval], label='UG-BEACON', marker='^', markersize=marker_size,linewidth=linewidth)
# plt.plot(cost_NS_xspace_mean[::marker_interval], coverage_NS_xspace_mean[::marker_interval], label='NS-FS', marker='p', markersize=marker_size,color='mediumpurple',linewidth=linewidth)
axes[1].plot(cost_RS_mean[::marker_interval], coverage_RS_mean[::marker_interval], label='RS', marker='v', markersize=marker_size,color='hotpink',linewidth=linewidth)
axes[1].plot(cost_MaxVar_mean[::marker_interval], coverage_MaxVar_mean[::marker_interval], label='MaxVar', marker='s', markersize=marker_size,color='green',linewidth=linewidth)

axes[1].fill_between(cost_NS_mean_TS1, coverage_NS_mean_TS1 - coverage_NS_std_TS1, coverage_NS_mean_TS1 + coverage_NS_std_TS1,  alpha=0.3)
axes[1].fill_between(cost_BO_mean, coverage_BO_mean - coverage_BO_std, coverage_BO_mean + coverage_BO_std,  alpha=0.3)
# plt.fill_between(cost_NS_xspace_mean, coverage_NS_xspace_mean - coverage_NS_xspace_std, coverage_NS_xspace_mean + coverage_NS_xspace_std,  alpha=0.3,color='mediumpurple')
axes[1].fill_between(cost_RS_mean, coverage_RS_mean - coverage_RS_std, coverage_RS_mean + coverage_RS_std,  alpha=0.3,color='hotpink')
axes[1].fill_between(cost_MaxVar_mean, coverage_MaxVar_mean - coverage_MaxVar_std, coverage_MaxVar_mean + coverage_MaxVar_std,  alpha=0.3,color='green')


axes[1].set_xlabel('Number of evaluations', fontsize='large')
axes[1].set_ylabel('Reachability', fontsize='large')
axes[1].legend(prop={'weight':'normal','size':'medium'}, loc='best')

# axes[1].plot(
#     cost_NS_mean_TS1[::marker_interval],
#     coverage_NS_mean_TS1[::marker_interval],
#     label='BEACON',
#     marker='X',
#     markersize=marker_size,
#     linewidth=linewidth,
# )
# axes[1].set_title('Cost vs Coverage')
# axes[1].set_xlabel('Cost')
# axes[1].set_ylabel('Coverage')
# axes[1].legend()



# Adjust layout and display
# plt.tight_layout()
# plt.show()
