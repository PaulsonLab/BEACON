#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 16 14:38:55 2024

@author: tang.1856
"""
import torch
import matplotlib.pyplot as plt

synthetic = '4DAckley'

cost_NS_TS1 = torch.load('/home/tang.1856/Jonathan/Novelty Search/Continuous_Synthetic_Code/Results/'+synthetic+'/'+synthetic+'_TS_1_cost_list_NS.pt')
coverage_NS_TS1 = torch.load('/home/tang.1856/Jonathan/Novelty Search/Continuous_Synthetic_Code/Results/'+synthetic+'/'+synthetic+'_TS_1_coverage_list_NS.pt')

cost_BO = torch.load('/home/tang.1856/Jonathan/Novelty Search/Continuous_Synthetic_Code/Results/'+synthetic+'/'+synthetic+'_cost_list_MaxVar.pt')
coverage_BO = torch.load('/home/tang.1856/Jonathan/Novelty Search/Continuous_Synthetic_Code/Results/'+synthetic+'/'+synthetic+'_coverage_list_MaxVar.pt')

cost_RS = torch.load('/home/tang.1856/Jonathan/Novelty Search/Continuous_Synthetic_Code/Results/'+synthetic+'/'+synthetic+'_cost_list_RS.pt')
coverage_RS = torch.load('/home/tang.1856/Jonathan/Novelty Search/Continuous_Synthetic_Code/Results/'+synthetic+'/'+synthetic+'_coverage_list_RS.pt')

cost_NS_mean = torch.load('/home/tang.1856/Jonathan/Novelty Search/Continuous_Synthetic_Code/Results/'+synthetic+'/'+synthetic+'_cost_list_NS_mean.pt')
coverage_NS_mean = torch.load('/home/tang.1856/Jonathan/Novelty Search/Continuous_Synthetic_Code/Results/'+synthetic+'/'+synthetic+'_coverage_list_NS_mean.pt')

cost_GA_NS_random = torch.load('/home/tang.1856/Jonathan/Novelty Search/Continuous_Synthetic_Code/Results/'+synthetic+'/'+synthetic+'_cost_list_GA_random.pt')
coverage_GA_NS_random = torch.load('/home/tang.1856/Jonathan/Novelty Search/Continuous_Synthetic_Code/Results/'+synthetic+'/'+synthetic+'_coverage_list_GA_random.pt')

cost_GA_NS_novel = torch.load('/home/tang.1856/Jonathan/Novelty Search/Continuous_Synthetic_Code/Results/'+synthetic+'/'+synthetic+'_cost_list_GA_novel.pt')
coverage_GA_NS_novel = torch.load('/home/tang.1856/Jonathan/Novelty Search/Continuous_Synthetic_Code/Results/'+synthetic+'/'+synthetic+'_coverage_list_GA_novel.pt')

cost_DEA = torch.load('/home/tang.1856/Jonathan/Novelty Search/Continuous_Synthetic_Code/Results/'+synthetic+'/'+synthetic+'_cost_list_GA_DEA.pt')
coverage_DEA = torch.load('/home/tang.1856/Jonathan/Novelty Search/Continuous_Synthetic_Code/Results/'+synthetic+'/'+synthetic+'_coverage_list_GA_DEA.pt')

cost_sobol = torch.load('/home/tang.1856/Jonathan/Novelty Search/Continuous_Synthetic_Code/Results/'+synthetic+'/'+synthetic+'_cost_list_sobol.pt')
coverage_sobol = torch.load('/home/tang.1856/Jonathan/Novelty Search/Continuous_Synthetic_Code/Results/'+synthetic+'/'+synthetic+'_coverage_list_sobol.pt')

cost_NS_xspace = torch.load('/home/tang.1856/Jonathan/Novelty Search/Continuous_Synthetic_Code/Results/'+synthetic+'/'+synthetic+'_cost_list_NS_x_space.pt')
coverage_NS_xspace = torch.load('/home/tang.1856/Jonathan/Novelty Search/Continuous_Synthetic_Code/Results/'+synthetic+'/'+synthetic+'_coverage_list_NS_x_space.pt')



coverage_NS_mean_TS1 = torch.mean(coverage_NS_TS1, dim = 0)
coverage_NS_std_TS1 = torch.std(coverage_NS_TS1, dim = 0)
cost_NS_mean_TS1 = torch.mean(cost_NS_TS1, dim = 0)

coverage_BO_mean = torch.mean(coverage_BO, dim = 0)
coverage_BO_std = torch.std(coverage_BO, dim = 0)
cost_BO_mean = torch.mean(cost_BO, dim = 0)

coverage_RS_mean = torch.mean(coverage_RS, dim = 0)
coverage_RS_std = torch.std(coverage_RS, dim = 0)
cost_RS_mean = torch.mean(cost_RS, dim = 0)

coverage_NS_mean_mean = torch.mean(coverage_NS_mean, dim = 0)
coverage_NS_mean_std = torch.std(coverage_NS_mean, dim = 0)
cost_NS_mean_mean = torch.mean(cost_NS_mean, dim = 0)

coverage_GA_NS_random_mean = torch.mean(coverage_GA_NS_random , dim = 0)
coverage_GA_NS_random_std = torch.std(coverage_GA_NS_random , dim = 0)
cost_GA_NS_random_mean = torch.mean(cost_GA_NS_random , dim = 0)

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

text_size = 24
marker_size = 18
linewidth=4
marker_interval = 8
weight='bold'
alpha = 0.3

plt.figure(figsize=(16,14))
# plt.grid(color='lightgrey', linewidth=0.5)
plt.plot(cost_NS_mean_TS1[::marker_interval], coverage_NS_mean_TS1[::marker_interval], label='BEACON', marker='X', markersize=marker_size, linewidth=linewidth)
plt.plot(cost_BO_mean[::marker_interval], coverage_BO_mean[::marker_interval], label='MaxVar', marker='^', markersize=marker_size, linewidth=linewidth)
plt.plot(cost_GA_NS_novel_mean, coverage_GA_NS_novel_mean, label='NS-EA', marker='>', markersize=marker_size, linewidth=linewidth )
plt.plot(cost_DEA_mean, coverage_DEA_mean, label='NS-DEA', marker='D', markersize=marker_size, linewidth=linewidth )
plt.plot(cost_NS_xspace_mean[::marker_interval], coverage_NS_xspace_mean[::marker_interval], label='NS-FS', marker='p', markersize=marker_size, linewidth=linewidth )
plt.plot(cost_sobol_mean[::marker_interval], coverage_sobol_mean[::marker_interval], label='Sobol', marker='o', markersize=marker_size, linewidth=linewidth )
plt.plot(cost_RS_mean[::marker_interval], coverage_RS_mean[::marker_interval], label='RS', marker='v', markersize=marker_size, linewidth=linewidth )

plt.fill_between(cost_NS_mean_TS1, coverage_NS_mean_TS1 - coverage_NS_std_TS1, coverage_NS_mean_TS1 + coverage_NS_std_TS1,  alpha=alpha)
plt.fill_between(cost_BO_mean, coverage_BO_mean - coverage_BO_std, coverage_BO_mean + coverage_BO_std,  alpha=alpha)
plt.fill_between(cost_GA_NS_novel_mean, coverage_GA_NS_novel_mean - coverage_GA_NS_novel_std, coverage_GA_NS_novel_mean + coverage_GA_NS_novel_std,  alpha=alpha)
plt.fill_between(cost_DEA_mean, coverage_DEA_mean - coverage_DEA_std, coverage_DEA_mean + coverage_DEA_std,  alpha=alpha)
plt.fill_between(cost_NS_xspace_mean, coverage_NS_xspace_mean - coverage_NS_xspace_std, coverage_NS_xspace_mean + coverage_NS_xspace_std,  alpha=alpha)
plt.fill_between(cost_sobol_mean, coverage_sobol_mean - coverage_sobol_std, coverage_sobol_mean + coverage_sobol_std,  alpha=alpha)
plt.fill_between(cost_RS_mean, coverage_RS_mean - coverage_RS_std, coverage_RS_mean + coverage_RS_std,  alpha=alpha)

plt.xlabel('Number of evaluations', fontsize=text_size, fontweight=weight)
plt.ylabel('Reachability', fontsize=text_size, fontweight=weight)
plt.legend(prop={'weight':'bold','size':text_size})
# plt.title(synthetic)

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




