#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 16 14:38:55 2024

@author: tang.1856
"""
import torch
import matplotlib.pyplot as plt

synthetic = '4DAckley'
ablation = 'offspring'

cost_GA_NS_novel = torch.load('/home/tang.1856/Jonathan/Novelty Search/EA_NoveltySearch/Results/'+synthetic+'/'+synthetic+'_cost_list_GA_'+str(ablation)+'10.pt')
coverage_GA_NS_novel = torch.load('/home/tang.1856/Jonathan/Novelty Search/EA_NoveltySearch/Results/'+synthetic+'/'+synthetic+'_coverage_list_GA_'+str(ablation)+'10.pt')

cost_GA_NS_novel2 = torch.load('/home/tang.1856/Jonathan/Novelty Search/EA_NoveltySearch/Results/'+synthetic+'/'+synthetic+'_cost_list_GA_'+str(ablation)+'20.pt')
coverage_GA_NS_novel2 = torch.load('/home/tang.1856/Jonathan/Novelty Search/EA_NoveltySearch/Results/'+synthetic+'/'+synthetic+'_coverage_list_GA_'+str(ablation)+'20.pt')

cost_GA_NS_novel3 = torch.load('/home/tang.1856/Jonathan/Novelty Search/EA_NoveltySearch/Results/'+synthetic+'/'+synthetic+'_cost_list_GA_'+str(ablation)+'40.pt')
coverage_GA_NS_novel3 = torch.load('/home/tang.1856/Jonathan/Novelty Search/EA_NoveltySearch/Results/'+synthetic+'/'+synthetic+'_coverage_list_GA_'+str(ablation)+'40.pt')


coverage_GA_NS_novel_mean = torch.mean(coverage_GA_NS_novel, dim = 0)
coverage_GA_NS_novel_std = torch.std(coverage_GA_NS_novel , dim = 0)
cost_GA_NS_novel_mean = torch.mean(cost_GA_NS_novel , dim = 0)

coverage_GA_NS_novel_mean2 = torch.mean(coverage_GA_NS_novel2, dim = 0)
coverage_GA_NS_novel_std2 = torch.std(coverage_GA_NS_novel2 , dim = 0)
cost_GA_NS_novel_mean2 = torch.mean(cost_GA_NS_novel2 , dim = 0)

coverage_GA_NS_novel_mean3 = torch.mean(coverage_GA_NS_novel3, dim = 0)
coverage_GA_NS_novel_std3 = torch.std(coverage_GA_NS_novel3 , dim = 0)
cost_GA_NS_novel_mean3 = torch.mean(cost_GA_NS_novel3 , dim = 0)



text_size = 24
marker_size = 18
linewidth=4
marker_interval = 2
weight='bold'
alpha = 0.3

plt.figure(figsize=(16,14))

# plt.plot(cost_GA_NS_novel_mean[::marker_interval], coverage_GA_NS_novel_mean[::marker_interval], label='Population=10', marker='o', markersize=marker_size, linewidth=linewidth)
# plt.fill_between(cost_GA_NS_novel_mean, coverage_GA_NS_novel_mean - coverage_GA_NS_novel_std, coverage_GA_NS_novel_mean + coverage_GA_NS_novel_std,  alpha=0.3)

# plt.plot(cost_GA_NS_novel_mean2[::marker_interval], coverage_GA_NS_novel_mean2[::marker_interval], label='Population=20', marker='>', markersize=marker_size, linewidth=linewidth)
# plt.fill_between(cost_GA_NS_novel_mean2, coverage_GA_NS_novel_mean2 - coverage_GA_NS_novel_std2, coverage_GA_NS_novel_mean2 + coverage_GA_NS_novel_std2,  alpha=0.3)

# plt.plot(cost_GA_NS_novel_mean3[::marker_interval], coverage_GA_NS_novel_mean3[::marker_interval], label='Population=40', marker='s', markersize=marker_size, linewidth=linewidth)
# plt.fill_between(cost_GA_NS_novel_mean3, coverage_GA_NS_novel_mean3 - coverage_GA_NS_novel_std3, coverage_GA_NS_novel_mean3 + coverage_GA_NS_novel_std3,  alpha=0.3)

plt.plot(cost_GA_NS_novel_mean[::marker_interval], coverage_GA_NS_novel_mean[::marker_interval], label='Offspring=10', marker='o', markersize=marker_size, linewidth=linewidth)
plt.fill_between(cost_GA_NS_novel_mean, coverage_GA_NS_novel_mean - coverage_GA_NS_novel_std, coverage_GA_NS_novel_mean + coverage_GA_NS_novel_std,  alpha=0.3)

plt.plot(cost_GA_NS_novel_mean2[::int(marker_interval*0.5)], coverage_GA_NS_novel_mean2[::int(marker_interval*0.5)], label='Offspring=20', marker='>', markersize=marker_size, linewidth=linewidth)
plt.fill_between(cost_GA_NS_novel_mean2, coverage_GA_NS_novel_mean2 - coverage_GA_NS_novel_std2, coverage_GA_NS_novel_mean2 + coverage_GA_NS_novel_std2,  alpha=0.3)

plt.plot(cost_GA_NS_novel_mean3[::int(marker_interval*0.5)], coverage_GA_NS_novel_mean3[::int(marker_interval*0.5)], label='Offspring=40', marker='s', markersize=marker_size, linewidth=linewidth)
plt.fill_between(cost_GA_NS_novel_mean3, coverage_GA_NS_novel_mean3 - coverage_GA_NS_novel_std3, coverage_GA_NS_novel_mean3 + coverage_GA_NS_novel_std3,  alpha=0.3)

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
