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
time_list2, time_list_err2 =[], []

time_NS_TS1 = (torch.load('/home/tang.1856/Jonathan/Novelty Search/Continuous_Synthetic_Code/Results/Sparse/Time/20DRosen_time_list_NS_GPflow_sparse_time_100.pt'))
time_NS_TS2 = (torch.load('/home/tang.1856/Jonathan/Novelty Search/Continuous_Synthetic_Code/Results/Sparse/Time/20DRosen_time_list_NS_GPflow_sparse_time_500.pt'))
time_NS_TS3 = (torch.load('/home/tang.1856/Jonathan/Novelty Search/Continuous_Synthetic_Code/Results/Sparse/Time/20DRosen_time_list_NS_GPflow_sparse_time_1000.pt'))
time_NS_TS4 = (torch.load('/home/tang.1856/Jonathan/Novelty Search/Continuous_Synthetic_Code/Results/Sparse/Time/20DRosen_time_list_NS_GPflow_sparse_time_2000.pt'))
time_NS_TS9 = (torch.load('/home/tang.1856/Jonathan/Novelty Search/Continuous_Synthetic_Code/Results/Sparse/Time/20DRosen_time_list_NS_GPflow_sparse_time_3000.pt'))
time_NS_TS10 = (torch.load('/home/tang.1856/Jonathan/Novelty Search/Continuous_Synthetic_Code/Results/Sparse/Time/20DRosen_time_list_NS_GPflow_sparse_time_4000.pt'))
time_NS_TS11 = (torch.load('/home/tang.1856/Jonathan/Novelty Search/Continuous_Synthetic_Code/Results/Sparse/Time/20DRosen_time_list_NS_GPflow_sparse_time_5000.pt'))

time_NS_TS5 = (torch.load('/home/tang.1856/Jonathan/Novelty Search/Continuous_Synthetic_Code/Results/Sparse/Time/20DRosen_time_list_NS_GPflow_time_100.pt'))
time_NS_TS6 = (torch.load('/home/tang.1856/Jonathan/Novelty Search/Continuous_Synthetic_Code/Results/Sparse/Time/20DRosen_time_list_NS_GPflow_time_500.pt'))
time_NS_TS7 = (torch.load('/home/tang.1856/Jonathan/Novelty Search/Continuous_Synthetic_Code/Results/Sparse/Time/20DRosen_time_list_NS_GPflow_time_1000.pt'))
time_NS_TS8 = (torch.load('/home/tang.1856/Jonathan/Novelty Search/Continuous_Synthetic_Code/Results/Sparse/Time/20DRosen_time_list_NS_GPflow_time_2000.pt'))
time_NS_TS12 = (torch.load('/home/tang.1856/Jonathan/Novelty Search/Continuous_Synthetic_Code/Results/Sparse/Time/20DRosen_time_list_NS_GPflow_time_2500.pt'))


time_NS_mean_TS1 = (torch.mean(time_NS_TS1, dim = 0))
time_NS_std_TS1 = torch.std(time_NS_TS1, dim = 0)
time_list.append(time_NS_mean_TS1)
time_list_err.append(time_NS_std_TS1)


time_NS_mean_TS2 = torch.mean(time_NS_TS2, dim = 0)
time_NS_std_TS2 = torch.std(time_NS_TS2, dim = 0)
time_list.append(time_NS_mean_TS2)
time_list_err.append(time_NS_std_TS2)


time_NS_mean_TS3 = torch.mean(time_NS_TS3, dim = 0)
time_NS_std_TS3 = torch.std(time_NS_TS3, dim = 0)
time_list.append(time_NS_mean_TS3)
time_list_err.append(time_NS_std_TS3)

time_NS_mean_TS4 = torch.mean(time_NS_TS4, dim = 0)
time_NS_std_TS4 = torch.std(time_NS_TS4, dim = 0)
time_list.append(time_NS_mean_TS4)
time_list_err.append(time_NS_std_TS4)


time_NS_mean_TS5 = torch.mean(time_NS_TS5, dim = 0)
time_NS_std_TS5 = torch.std(time_NS_TS5, dim = 0)
time_list2.append(time_NS_mean_TS5)
time_list_err2.append(time_NS_std_TS5)


time_NS_mean_TS6 = torch.mean(time_NS_TS6, dim = 0)
time_NS_std_TS6 = torch.std(time_NS_TS6, dim = 0)
time_list2.append(time_NS_mean_TS6)
time_list_err2.append(time_NS_std_TS6)

time_NS_mean_TS7 = torch.mean(time_NS_TS7, dim = 0)
time_NS_std_TS7 = torch.std(time_NS_TS7, dim = 0)
time_list2.append(time_NS_mean_TS7)
time_list_err2.append(time_NS_std_TS7)

time_NS_mean_TS8 = torch.mean(time_NS_TS8, dim = 0)
time_NS_std_TS8 = torch.std(time_NS_TS8, dim = 0)
time_list2.append(time_NS_mean_TS8)
time_list_err2.append(time_NS_std_TS8)

time_NS_mean_TS9 = torch.mean(time_NS_TS9, dim = 0)
time_NS_std_TS9 = torch.std(time_NS_TS9, dim = 0)
time_list.append(time_NS_mean_TS9)
time_list_err.append(time_NS_std_TS9)

time_NS_mean_TS10 = torch.mean(time_NS_TS10, dim = 0)
time_NS_std_TS10 = torch.std(time_NS_TS10, dim = 0)
time_list.append(time_NS_mean_TS10)
time_list_err.append(time_NS_std_TS10)

time_NS_mean_TS11 = torch.mean(time_NS_TS11, dim = 0)
time_NS_std_TS11 = torch.std(time_NS_TS11, dim = 0)
time_list.append(time_NS_mean_TS11)
time_list_err.append(time_NS_std_TS11)

time_NS_mean_TS12 = torch.mean(time_NS_TS12, dim = 0)
time_NS_std_TS12 = torch.std(time_NS_TS12, dim = 0)
time_list2.append(time_NS_mean_TS12)
time_list_err2.append(time_NS_std_TS12)



# coverage_NS_mean_TS4 = torch.mean(coverage_NS_TS4, dim = 0)
# coverage_NS_std_TS4 = torch.std(coverage_NS_TS4, dim = 0)
# cost_NS_mean_TS4 = torch.mean(cost_NS_TS4, dim = 0)
xx1 = [100,500,1000,2000,3000,4000,5000]
xx2 = [100,500,1000,2000,2500]
text_size = 24
marker_size = 18
linewidth=4
marker_interval = 8
weight='bold'
alpha = 0.3
plt.figure(figsize=(12,10))
# plt.errorbar(x=xx1, y=time_list, yerr=time_list_err , fmt='o', capsize=5, linestyle='', color='b', ecolor='r', markersize=marker_size, linewidth=linewidth, label='SVGP')
# plt.errorbar(x=xx2, y=time_list2, yerr=time_list_err2 , fmt='o', capsize=5, linestyle='', color='g', ecolor='b', markersize=marker_size, linewidth=linewidth, label='GP')
# plt.xticks(xx1)
# plt.xlabel('N', fontsize=text_size, fontweight=weight)
# plt.ylabel('CPU time (sec)', fontsize=text_size, fontweight=weight)
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

# ax.set_xticklabels(xx1)


# plt.figure(figsize=(16,14))
# # ax = plt.axes()
# # ax.set_facecolor("lightgrey")
# plt.plot(cost_NS_mean_TS1[::marker_interval], coverage_NS_mean_TS1[::marker_interval], label='m=1', marker='X', markersize=marker_size, linewidth=linewidth)
# plt.fill_between(cost_NS_mean_TS1, coverage_NS_mean_TS1 - coverage_NS_std_TS1, coverage_NS_mean_TS1 + coverage_NS_std_TS1,  alpha=0.2)

# plt.plot(cost_NS_mean_TS2[::marker_interval], coverage_NS_mean_TS2[::marker_interval], label='m=5', marker='o', markersize=marker_size, linewidth=linewidth)
# plt.fill_between(cost_NS_mean_TS2, coverage_NS_mean_TS2 - coverage_NS_std_TS2, coverage_NS_mean_TS2 + coverage_NS_std_TS2,  alpha=0.2)

# plt.plot(cost_NS_mean_TS3[::marker_interval], coverage_NS_mean_TS3[::marker_interval], label='m=10', marker='^', markersize=marker_size, linewidth=linewidth)
# plt.fill_between(cost_NS_mean_TS3, coverage_NS_mean_TS3 - coverage_NS_std_TS3, coverage_NS_mean_TS3 + coverage_NS_std_TS3,  alpha=0.2)

# # plt.plot(cost_NS_mean_TS4[::marker_interval], coverage_NS_mean_TS4[::marker_interval], label='NS-BO(k=1)', marker='s', markersize=6)
# # plt.fill_between(cost_NS_mean_TS4, coverage_NS_mean_TS4 - coverage_NS_std_TS4, coverage_NS_mean_TS4 + coverage_NS_std_TS4,  alpha=0.3)

plt.semilogy(xx1, time_list, label='SVGP', marker='X', markersize=marker_size, linewidth=linewidth)
plt.semilogy(xx2, time_list2, label='GP', marker='s', markersize=marker_size, linewidth=linewidth)
plt.xlabel('number of training data points', fontsize=text_size, fontweight=weight)
plt.ylabel('CPU time (seconds)', fontsize=text_size, fontweight=weight)
plt.legend(prop={'weight':'bold','size':text_size})
# # plt.title(synthetic)

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



