#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 16 14:38:55 2024

@author: tang.1856
"""
import torch
import matplotlib.pyplot as plt

synthetic = '4DAckley'
bins = 1
path = '/home/tang.1856/Jonathan/Novelty Search/Continuous_Synthetic_Code/Results/Different TS/'

time_list, time_list_err =[], []
cost_NS_TS1 = torch.load(path+synthetic+'/'+synthetic+'_cost_list_NS_'+'TS'+str(bins)+'.pt')
coverage_NS_TS1 = torch.load(path+synthetic+'/'+synthetic+'_coverage_list_NS_'+'TS'+str(bins)+'.pt')
time_NS_TS1 = torch.load(path+synthetic+'/'+synthetic+'_time_list_NS_'+'TS'+str(bins)+'.pt')

cost_NS_TS2 = torch.load(path+synthetic+'/'+synthetic+'_cost_list_NS_'+'TS'+str(bins*5)+'.pt')
coverage_NS_TS2 = torch.load(path+synthetic+'/'+synthetic+'_coverage_list_NS_'+'TS'+str(bins*5)+'.pt')
time_NS_TS2 = torch.load(path+synthetic+'/'+synthetic+'_time_list_NS_'+'TS'+str(bins*5)+'.pt')

cost_NS_TS3 = torch.load(path+synthetic+'/'+synthetic+'_cost_list_NS_'+'TS'+str(bins*10)+'.pt')
coverage_NS_TS3 = torch.load(path+synthetic+'/'+synthetic+'_coverage_list_NS_'+'TS'+str(bins*10)+'.pt')
time_NS_TS3 = torch.load(path+synthetic+'/'+synthetic+'_time_list_NS_'+'TS'+str(bins*10)+'.pt')

coverage_NS_mean_TS1 = torch.mean(coverage_NS_TS1, dim = 0)
coverage_NS_std_TS1 = torch.std(coverage_NS_TS1, dim = 0)
cost_NS_mean_TS1 = torch.mean(cost_NS_TS1, dim = 0)
time_NS_mean_TS1 = torch.mean(time_NS_TS1, dim = 0)
time_NS_std_TS1 = torch.std(time_NS_TS1, dim = 0)
time_list.append(time_NS_mean_TS1)
time_list_err.append(time_NS_std_TS1)

coverage_NS_mean_TS2 = torch.mean(coverage_NS_TS2, dim = 0)
coverage_NS_std_TS2 = torch.std(coverage_NS_TS2, dim = 0)
cost_NS_mean_TS2 = torch.mean(cost_NS_TS2, dim = 0)
time_NS_mean_TS2 = torch.mean(time_NS_TS2, dim = 0)
time_NS_std_TS2 = torch.std(time_NS_TS2, dim = 0)
time_list.append(time_NS_mean_TS2)
time_list_err.append(time_NS_std_TS2)

coverage_NS_mean_TS3 = torch.mean(coverage_NS_TS3, dim = 0)
coverage_NS_std_TS3 = torch.std(coverage_NS_TS3, dim = 0)
cost_NS_mean_TS3 = torch.mean(cost_NS_TS3, dim = 0)
time_NS_mean_TS3 = torch.mean(time_NS_TS3, dim = 0)
time_NS_std_TS3 = torch.std(time_NS_TS3, dim = 0)
time_list.append(time_NS_mean_TS3)
time_list_err.append(time_NS_std_TS3)

# coverage_NS_mean_TS4 = torch.mean(coverage_NS_TS4, dim = 0)
# coverage_NS_std_TS4 = torch.std(coverage_NS_TS4, dim = 0)
# cost_NS_mean_TS4 = torch.mean(cost_NS_TS4, dim = 0)
xx = [1,5,10]

text_size = 24
marker_size = 18
linewidth=4
marker_interval = 8
weight='bold'
alpha = 0.3
plt.figure(figsize=(16,14))
plt.errorbar(x=xx, y=time_list, yerr=time_list_err , fmt='o', capsize=5, linestyle='', color='b', ecolor='r', markersize=marker_size, linewidth=linewidth)
plt.xticks(xx)
plt.xlabel('m', fontsize=text_size, fontweight=weight)
plt.ylabel('CPU time per iteration (sec)', fontsize=text_size, fontweight=weight)
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




plt.figure(figsize=(16,14))
# ax = plt.axes()
# ax.set_facecolor("lightgrey")
plt.plot(cost_NS_mean_TS1[::marker_interval], coverage_NS_mean_TS1[::marker_interval], label='m=1', marker='X', markersize=marker_size, linewidth=linewidth)
plt.fill_between(cost_NS_mean_TS1, coverage_NS_mean_TS1 - coverage_NS_std_TS1, coverage_NS_mean_TS1 + coverage_NS_std_TS1,  alpha=0.2)

plt.plot(cost_NS_mean_TS2[::marker_interval], coverage_NS_mean_TS2[::marker_interval], label='m=5', marker='o', markersize=marker_size, linewidth=linewidth)
plt.fill_between(cost_NS_mean_TS2, coverage_NS_mean_TS2 - coverage_NS_std_TS2, coverage_NS_mean_TS2 + coverage_NS_std_TS2,  alpha=0.2)

plt.plot(cost_NS_mean_TS3[::marker_interval], coverage_NS_mean_TS3[::marker_interval], label='m=10', marker='^', markersize=marker_size, linewidth=linewidth)
plt.fill_between(cost_NS_mean_TS3, coverage_NS_mean_TS3 - coverage_NS_std_TS3, coverage_NS_mean_TS3 + coverage_NS_std_TS3,  alpha=0.2)

# plt.plot(cost_NS_mean_TS4[::marker_interval], coverage_NS_mean_TS4[::marker_interval], label='NS-BO(k=1)', marker='s', markersize=6)
# plt.fill_between(cost_NS_mean_TS4, coverage_NS_mean_TS4 - coverage_NS_std_TS4, coverage_NS_mean_TS4 + coverage_NS_std_TS4,  alpha=0.3)

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
