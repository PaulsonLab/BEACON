#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  2 09:07:23 2024

@author: tang.1856
"""

import torch
import matplotlib.pyplot as plt

synthetic = 'DNA'
cost_NS_TS1 = torch.load('/home/tang.1856/BEACON/BEACON/Discrete_Material_Code/Results/'+synthetic+'/'+synthetic+'_TS_1_cost_list_NS1.pt')
coverage_NS_TS1 = torch.load('/home/tang.1856/BEACON/BEACON/Discrete_Material_Code/Results/'+synthetic+'/'+synthetic+'_TS_1_coverage_list_NS1.pt')

cost_BO = torch.load('/home/tang.1856/BEACON/BEACON/Discrete_Material_Code/Results/'+synthetic+'/'+synthetic+'_cost_list_MaxVar1.pt')
coverage_BO = torch.load('/home/tang.1856/BEACON/BEACON/Discrete_Material_Code/Results/'+synthetic+'/'+synthetic+'_coverage_list_MaxVar1.pt')

cost_RS = torch.load('/home/tang.1856/BEACON/BEACON/Discrete_Material_Code/Results/'+synthetic+'/'+synthetic+'_cost_list_RS1.pt')
coverage_RS = torch.load('/home/tang.1856/BEACON/BEACON/Discrete_Material_Code/Results/'+synthetic+'/'+synthetic+'_coverage_list_RS1.pt')

cost_NS_xspace = torch.load('/home/tang.1856/BEACON/BEACON/Discrete_Material_Code/Results/'+synthetic+'/'+synthetic+'_cost_list_NS_xspace.pt')
coverage_NS_xspace = torch.load('/home/tang.1856/BEACON/BEACON/Discrete_Material_Code/Results/'+synthetic+'/'+synthetic+'_coverage_list_NS_xspace.pt')

cost_EI = torch.load('/home/tang.1856/BEACON/BEACON/Discrete_Material_Code/Results/'+synthetic+'/'+synthetic+'_cost_list_EI.pt')
coverage_EI = torch.load('/home/tang.1856/BEACON/BEACON/Discrete_Material_Code/Results/'+synthetic+'/'+synthetic+'_coverage_list_EI.pt')

cost_CMAES = torch.load('/home/tang.1856/BEACON/BEACON/Discrete_Material_Code/Results/'+synthetic+'/'+synthetic+'_cost_list_CMAES1.pt').to(torch.float)
coverage_CMAES = torch.load('/home/tang.1856/BEACON/BEACON/Discrete_Material_Code/Results/'+synthetic+'/'+synthetic+'_coverage_list_CMAES1.pt')


coverage_NS_mean_TS1 = torch.mean(coverage_NS_TS1, dim = 0)
coverage_NS_std_TS1 = torch.std(coverage_NS_TS1, dim = 0)
cost_NS_mean_TS1 = torch.mean(cost_NS_TS1, dim = 0)

coverage_BO_mean = torch.mean(coverage_BO, dim = 0)
coverage_BO_std = torch.std(coverage_BO, dim = 0)
cost_BO_mean = torch.mean(cost_BO, dim = 0)

coverage_RS_mean = torch.mean(coverage_RS, dim = 0)
coverage_RS_std = torch.std(coverage_RS, dim = 0)
cost_RS_mean = torch.mean(cost_RS, dim = 0)

coverage_NS_xspace_mean = torch.mean(coverage_NS_xspace, dim = 0)
coverage_NS_xspace_std = torch.std(coverage_NS_xspace , dim = 0)
cost_NS_xspace_mean = torch.mean(cost_NS_xspace , dim = 0)

coverage_EI_mean = torch.mean(coverage_EI, dim = 0)
coverage_EI_std = torch.std(coverage_EI , dim = 0)
cost_EI_mean = torch.mean(cost_EI , dim = 0)

coverage_CMAES_mean = torch.mean(coverage_CMAES, dim = 0)
coverage_CMAES_std = torch.std(coverage_CMAES , dim = 0)
cost_CMAES_mean = torch.mean(cost_CMAES , dim = 0)

text_size = 24
marker_size = 18
linewidth=4
marker_interval = 10
weight='bold'
alpha = 0.3
plt.figure(figsize=(12,10))
plt.plot(cost_NS_mean_TS1[::marker_interval], coverage_NS_mean_TS1[::marker_interval], label='BEACON', marker='X', markersize=marker_size, linewidth=linewidth)
plt.plot(cost_BO_mean[::marker_interval], coverage_BO_mean[::marker_interval], label='MaxVar', marker='^', markersize=marker_size, linewidth=linewidth)
plt.plot(cost_NS_xspace_mean[::marker_interval], coverage_NS_xspace_mean[::marker_interval], label='NS-FS', marker='p', markersize=marker_size,color='mediumpurple', linewidth=linewidth)
plt.plot(cost_RS_mean[::marker_interval], coverage_RS_mean[::marker_interval], label='RS', marker='v', markersize=marker_size, color='hotpink', linewidth=linewidth)
plt.plot(cost_EI_mean[::marker_interval], coverage_EI_mean[::marker_interval], label='EI', markersize=marker_size, marker='o', linewidth=linewidth)
plt.plot(cost_CMAES_mean, coverage_CMAES_mean, label='CMAES', markersize=marker_size, marker='o', linewidth=linewidth)

plt.fill_between(cost_NS_mean_TS1, coverage_NS_mean_TS1 - coverage_NS_std_TS1, coverage_NS_mean_TS1 + coverage_NS_std_TS1,  alpha=alpha)
plt.fill_between(cost_BO_mean, coverage_BO_mean - coverage_BO_std, coverage_BO_mean + coverage_BO_std,  alpha=alpha)
plt.fill_between(cost_NS_xspace_mean, coverage_NS_xspace_mean - coverage_NS_xspace_std, coverage_NS_xspace_mean + coverage_NS_xspace_std,  alpha=alpha,color='mediumpurple')
plt.fill_between(cost_RS_mean, coverage_RS_mean - coverage_RS_std, coverage_RS_mean + coverage_RS_std,  alpha=alpha, color='hotpink')
plt.fill_between(cost_EI_mean, coverage_EI_mean - coverage_EI_std, coverage_EI_mean + coverage_EI_std,  alpha=alpha)
plt.fill_between(cost_CMAES_mean, coverage_CMAES_mean - coverage_CMAES_std, coverage_CMAES_mean + coverage_CMAES_std,  alpha=alpha)

plt.xlabel('Number of evaluations', fontsize=text_size, fontweight=weight)
plt.ylabel('Reachability', fontsize=text_size, fontweight=weight)
plt.legend(prop={'weight':'bold','size':text_size})
plt.title('DNA TF Bind 8', fontsize=text_size, fontweight=weight)

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