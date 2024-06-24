# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# """
# Created on Sat Mar 16 14:38:55 2024

# @author: tang.1856
# """
# import torch
# import matplotlib.pyplot as plt

# synthetic = 'Maze'

# cost_NS_TS1 = torch.load('/home/tang.1856/Jonathan/Novelty Search/Continuous_MultiOutcome/Results/'+synthetic+'/'+synthetic+'_TS_1_cost_list_NS.pt')
# coverage_NS_TS1 = torch.load('/home/tang.1856/Jonathan/Novelty Search/Continuous_MultiOutcome/Results/'+synthetic+'/'+synthetic+'_TS_1_cumbent_list_NS2.pt')

# cost_BO = torch.load('/home/tang.1856/Jonathan/Novelty Search/Continuous_MultiOutcome/Results/'+synthetic+'/'+synthetic+'_cost_list_EI.pt')
# coverage_BO = torch.load('/home/tang.1856/Jonathan/Novelty Search/Continuous_MultiOutcome/Results/'+synthetic+'/'+synthetic+'_cumbent_list_EI.pt')

# cost_MaxVar = torch.load('/home/tang.1856/Jonathan/Novelty Search/Continuous_MultiOutcome/Results/'+synthetic+'/'+synthetic+'_cost_list_MaxVar.pt')
# coverage_MaxVar = torch.load('/home/tang.1856/Jonathan/Novelty Search/Continuous_MultiOutcome/Results/'+synthetic+'/'+synthetic+'_cumbent_list_MaxVar.pt')

# cost_RS = torch.load('/home/tang.1856/Jonathan/Novelty Search/Continuous_MultiOutcome/Results/'+synthetic+'/'+synthetic+'_cost_list_RS.pt')
# coverage_RS = torch.load('/home/tang.1856/Jonathan/Novelty Search/Continuous_MultiOutcome/Results/'+synthetic+'/'+synthetic+'_cumbent_list_RS.pt')

# cost_GA_NS_novel = torch.load('/home/tang.1856/Jonathan/Novelty Search/Continuous_MultiOutcome/Results/'+synthetic+'/'+synthetic+'_cost_list_GA.pt')
# coverage_GA_NS_novel = torch.load('/home/tang.1856/Jonathan/Novelty Search/Continuous_MultiOutcome/Results/'+synthetic+'/'+synthetic+'_cumbent_list_GA.pt')

# cost_DEA = torch.load('/home/tang.1856/Jonathan/Novelty Search/Continuous_MultiOutcome/Results/'+synthetic+'/'+synthetic+'_cost_list_DEA.pt')
# coverage_DEA = torch.load('/home/tang.1856/Jonathan/Novelty Search/Continuous_MultiOutcome/Results/'+synthetic+'/'+synthetic+'_cumbent_list_DEA.pt')

# cost_NS_xspace = torch.load('/home/tang.1856/Jonathan/Novelty Search/Continuous_MultiOutcome/Results/'+synthetic+'/'+synthetic+'_cost_list_NS_x_space.pt')
# coverage_NS_xspace = torch.load('/home/tang.1856/Jonathan/Novelty Search/Continuous_MultiOutcome/Results/'+synthetic+'/'+synthetic+'_cumbent_list_NS_x_space.pt')

# cost_sobol = torch.load('/home/tang.1856/Jonathan/Novelty Search/Continuous_MultiOutcome/Results/'+synthetic+'/'+synthetic+'_cost_list_sobol.pt')
# coverage_sobol = torch.load('/home/tang.1856/Jonathan/Novelty Search/Continuous_MultiOutcome/Results/'+synthetic+'/'+synthetic+'_cumbent_list_sobol.pt')

# coverage_NS_mean_TS1 = torch.mean(coverage_NS_TS1, dim = 0)
# coverage_NS_std_TS1 = torch.std(coverage_NS_TS1, dim = 0)
# cost_NS_mean_TS1 = torch.mean(cost_NS_TS1, dim = 0)


# coverage_BO_mean = torch.mean(coverage_BO, dim = 0)
# coverage_BO_std = torch.std(coverage_BO, dim = 0)
# cost_BO_mean = torch.mean(cost_BO, dim = 0)

# coverage_MaxVar_mean = torch.mean(coverage_MaxVar, dim = 0)
# coverage_MaxVar_std = torch.std(coverage_MaxVar, dim = 0)
# cost_MaxVar_mean = torch.mean(cost_MaxVar, dim = 0)

# coverage_RS_mean = torch.mean(coverage_RS, dim = 0)
# coverage_RS_std = torch.std(coverage_RS, dim = 0)
# cost_RS_mean = torch.mean(cost_RS, dim = 0)


# coverage_GA_NS_novel_mean = torch.mean(coverage_GA_NS_novel, dim = 0)
# coverage_GA_NS_novel_std = torch.std(coverage_GA_NS_novel , dim = 0)
# cost_GA_NS_novel_mean = torch.mean(cost_GA_NS_novel , dim = 0)

# coverage_DEA_mean = torch.mean(coverage_DEA, dim = 0)
# coverage_DEA_std = torch.std(coverage_DEA , dim = 0)
# cost_DEA_mean = torch.mean(cost_DEA , dim = 0)

# coverage_NS_xspace_mean = torch.mean(coverage_NS_xspace, dim = 0)
# coverage_NS_xspace_std = torch.std(coverage_NS_xspace , dim = 0)
# cost_NS_xspace_mean = torch.mean(cost_NS_xspace , dim = 0)

# coverage_sobol_mean = torch.mean(coverage_sobol, dim = 0)
# coverage_sobol_std = torch.std(coverage_sobol , dim = 0)
# cost_sobol_mean = torch.mean(cost_sobol , dim = 0)

# marker_interval = 5
# text_size = 24
# marker_size = 18
# weight='bold'
# linewidth = 4
# alpha = 0.3
# plt.figure(figsize=(16,14))

# plt.plot(cost_NS_mean_TS1[::marker_interval], coverage_NS_mean_TS1[::marker_interval], label='BEACON', marker='X', markersize=marker_size, linewidth=linewidth)
# plt.plot(cost_MaxVar_mean[::marker_interval], coverage_MaxVar_mean[::marker_interval], label='MaxVar', marker='^', markersize=marker_size, linewidth=linewidth)
# plt.plot(cost_GA_NS_novel_mean, coverage_GA_NS_novel_mean, label='NS-EA', marker='>', markersize=marker_size, linewidth=linewidth)
# plt.plot(cost_DEA_mean, coverage_DEA_mean, label='NS-DEA', marker='D', markersize=marker_size, linewidth=linewidth)
# plt.plot(cost_NS_xspace_mean[::marker_interval], coverage_NS_xspace_mean[::marker_interval], label='NS-EA-FS', marker='p', markersize=marker_size, linewidth=linewidth)
# plt.plot(cost_sobol_mean[::marker_interval], coverage_sobol_mean[::marker_interval], label='Sobol', marker='o', markersize=marker_size, linewidth=linewidth)
# plt.plot(cost_RS_mean[::marker_interval], coverage_RS_mean[::marker_interval], label='RS', marker='v', markersize=marker_size, linewidth=linewidth)
# plt.plot(cost_BO_mean[::marker_interval], coverage_BO_mean[::marker_interval], label='EI', marker='<', markersize=marker_size, linewidth=linewidth)

# # plt.fill_between(cost_NS_mean_TS1, coverage_NS_mean_TS1 - coverage_NS_std_TS1, coverage_NS_mean_TS1 + coverage_NS_std_TS1,  alpha=alpha )
# # plt.fill_between(cost_MaxVar_mean, coverage_MaxVar_mean - coverage_MaxVar_std, coverage_MaxVar_mean + coverage_MaxVar_std,  alpha=alpha )
# # plt.fill_between(cost_GA_NS_novel_mean, coverage_GA_NS_novel_mean - coverage_GA_NS_novel_std, coverage_GA_NS_novel_mean + coverage_GA_NS_novel_std,  alpha=alpha )
# # plt.fill_between(cost_DEA_mean, coverage_DEA_mean - coverage_DEA_std, coverage_DEA_mean + coverage_DEA_std,  alpha=alpha )
# # plt.fill_between(cost_NS_xspace_mean, coverage_NS_xspace_mean - coverage_NS_xspace_std, coverage_NS_xspace_mean + coverage_NS_xspace_std,  alpha=alpha )
# # plt.fill_between(cost_sobol_mean, coverage_sobol_mean - coverage_sobol_std, coverage_sobol_mean + coverage_sobol_std,  alpha=alpha )
# # plt.fill_between(cost_RS_mean, coverage_RS_mean - coverage_RS_std, coverage_RS_mean + coverage_RS_std,  alpha=alpha )
# # plt.fill_between(cost_BO_mean, coverage_BO_mean - coverage_BO_std, coverage_BO_mean + coverage_BO_std,  alpha=alpha )

# plt.ylim(0.48,1.02)
# plt.xlabel('Number of evaluations', fontsize=text_size, fontweight=weight)
# plt.ylabel('Best Reward', fontsize=text_size, fontweight=weight)
# plt.legend(prop={'weight':'bold','size':text_size}, loc=2)

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

# plt.grid(alpha=0.5, linewidth=2.0)



# import matplotlib.pyplot as plt
# import seaborn as sns
# import pandas as pd
# import numpy as np
# import torch

# # Example tensors (10 tensors for demonstration)
# tensors = {
#     'BEACON':coverage_NS_TS1[:,-1],
#     'MaxVar':coverage_MaxVar[:,-1],
#     'NS-EA':coverage_GA_NS_novel[:,-1],
#     'NS-DEA':coverage_DEA[:,-1],
#     'NS-EA-FS':coverage_NS_xspace[:,-1],
#     'Sobol':coverage_sobol[:,-1],
#     'RS':coverage_RS[:,-1],
#     'EI':coverage_BO[:,-1],       
# }

# # Convert tensors to lists and create a DataFrame
# data = {name: tensor.tolist() for name, tensor in tensors.items()}
# df = pd.DataFrame(data)

# # Melt the DataFrame to long format
# df_melted = df.melt(var_name='Tensor', value_name='Value')

# # Create the violin plot
# plt.figure(figsize=(18, 16))
# sns.violinplot(x='Tensor', y='Value', data=df_melted, palette='rainbow', width=0.4, inner='point', alpha=0.6)
# plt.ylabel('Final Value', fontsize=text_size, fontweight='bold')
# plt.xlabel('')
# plt.xticks(rotation=45, fontsize=12, fontweight='bold')
# plt.yticks(fontsize=12, fontweight='bold')
# # plt.title('Violin Plot of Tensors', fontsize=16, fontweight='bold')
# plt.grid(alpha=0.5, linewidth=1.0)

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
# plt.show()



import torch
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# Load your tensors (replace with actual data loading)
synthetic = 'Maze'

cost_NS_TS1 = torch.load('/home/tang.1856/Jonathan/Novelty Search/Continuous_MultiOutcome/Results/'+synthetic+'/'+synthetic+'_TS_1_cost_list_NS.pt')
coverage_NS_TS1 = torch.load('/home/tang.1856/Jonathan/Novelty Search/Continuous_MultiOutcome/Results/'+synthetic+'/'+synthetic+'_TS_1_cumbent_list_NS2.pt')

cost_BO = torch.load('/home/tang.1856/Jonathan/Novelty Search/Continuous_MultiOutcome/Results/'+synthetic+'/'+synthetic+'_cost_list_EI.pt')
coverage_BO = torch.load('/home/tang.1856/Jonathan/Novelty Search/Continuous_MultiOutcome/Results/'+synthetic+'/'+synthetic+'_cumbent_list_EI.pt')

cost_MaxVar = torch.load('/home/tang.1856/Jonathan/Novelty Search/Continuous_MultiOutcome/Results/'+synthetic+'/'+synthetic+'_cost_list_MaxVar.pt')
coverage_MaxVar = torch.load('/home/tang.1856/Jonathan/Novelty Search/Continuous_MultiOutcome/Results/'+synthetic+'/'+synthetic+'_cumbent_list_MaxVar.pt')

cost_RS = torch.load('/home/tang.1856/Jonathan/Novelty Search/Continuous_MultiOutcome/Results/'+synthetic+'/'+synthetic+'_cost_list_RS.pt')
coverage_RS = torch.load('/home/tang.1856/Jonathan/Novelty Search/Continuous_MultiOutcome/Results/'+synthetic+'/'+synthetic+'_cumbent_list_RS.pt')

cost_GA_NS_novel = torch.load('/home/tang.1856/Jonathan/Novelty Search/Continuous_MultiOutcome/Results/'+synthetic+'/'+synthetic+'_cost_list_GA.pt')
coverage_GA_NS_novel = torch.load('/home/tang.1856/Jonathan/Novelty Search/Continuous_MultiOutcome/Results/'+synthetic+'/'+synthetic+'_cumbent_list_GA.pt')

cost_DEA = torch.load('/home/tang.1856/Jonathan/Novelty Search/Continuous_MultiOutcome/Results/'+synthetic+'/'+synthetic+'_cost_list_DEA.pt')
coverage_DEA = torch.load('/home/tang.1856/Jonathan/Novelty Search/Continuous_MultiOutcome/Results/'+synthetic+'/'+synthetic+'_cumbent_list_DEA.pt')

cost_NS_xspace = torch.load('/home/tang.1856/Jonathan/Novelty Search/Continuous_MultiOutcome/Results/'+synthetic+'/'+synthetic+'_cost_list_NS_x_space.pt')
coverage_NS_xspace = torch.load('/home/tang.1856/Jonathan/Novelty Search/Continuous_MultiOutcome/Results/'+synthetic+'/'+synthetic+'_cumbent_list_NS_x_space.pt')

cost_sobol = torch.load('/home/tang.1856/Jonathan/Novelty Search/Continuous_MultiOutcome/Results/'+synthetic+'/'+synthetic+'_cost_list_sobol.pt')
coverage_sobol = torch.load('/home/tang.1856/Jonathan/Novelty Search/Continuous_MultiOutcome/Results/'+synthetic+'/'+synthetic+'_cumbent_list_sobol.pt')

# Define a consistent color palette
palette = {
    'BEACON': '#1f77b4',
    'MaxVar': '#ff7f0e',
    'NS-EA': '#2ca02c',
    'NS-DEA': '#d62728',
    'NS-FS': '#9467bd',
    'Sobol': '#8c564b',
    'RS': '#e377c2',
    'EI': '#7f7f7f'
}

tensors = {
    'BEACON':coverage_NS_TS1[:,-1],
    'MaxVar':coverage_MaxVar[:,-1],
    'NS-EA':coverage_GA_NS_novel[:,-1],
    'NS-DEA':coverage_DEA[:,-1],
    'NS-FS':coverage_NS_xspace[:,-1],
    'Sobol':coverage_sobol[:,-1],
    'RS':coverage_RS[:,-1],
    'EI':coverage_BO[:,-1],       
}

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

coverage_NS_xspace_mean = torch.mean(coverage_NS_xspace, dim=0)
coverage_NS_xspace_std = torch.std(coverage_NS_xspace, dim=0)
cost_NS_xspace_mean = torch.mean(cost_NS_xspace, dim=0)

coverage_sobol_mean = torch.mean(coverage_sobol, dim=0)
coverage_sobol_std = torch.std(coverage_sobol, dim=0)
cost_sobol_mean = torch.mean(cost_sobol, dim=0)

marker_interval = 5
text_size = 22
marker_size = 18
weight='bold'
linewidth = 4
alpha = 0.3

# Line plot
plt.figure(figsize=(12, 10))

plt.plot(cost_NS_mean_TS1[::marker_interval], coverage_NS_mean_TS1[::marker_interval], label='BEACON', marker='X', markersize=marker_size, linewidth=linewidth, color=palette['BEACON'])
plt.plot(cost_MaxVar_mean[::marker_interval], coverage_MaxVar_mean[::marker_interval], label='MaxVar', marker='^', markersize=marker_size, linewidth=linewidth, color=palette['MaxVar'])
plt.plot(cost_GA_NS_novel_mean, coverage_GA_NS_novel_mean, label='NS-EA', marker='>', markersize=marker_size, linewidth=linewidth, color=palette['NS-EA'])
plt.plot(cost_DEA_mean, coverage_DEA_mean, label='NS-DEA', marker='D', markersize=marker_size, linewidth=linewidth, color=palette['NS-DEA'])
plt.plot(cost_NS_xspace_mean[::marker_interval], coverage_NS_xspace_mean[::marker_interval], label='NS-FS', marker='p', markersize=marker_size, linewidth=linewidth, color=palette['NS-FS'])
plt.plot(cost_sobol_mean[::marker_interval], coverage_sobol_mean[::marker_interval], label='Sobol', marker='o', markersize=marker_size, linewidth=linewidth, color=palette['Sobol'])
plt.plot(cost_RS_mean[::marker_interval], coverage_RS_mean[::marker_interval], label='RS', marker='v', markersize=marker_size, linewidth=linewidth, color=palette['RS'])
plt.plot(cost_BO_mean[::marker_interval], coverage_BO_mean[::marker_interval], label='EI', marker='<', markersize=marker_size, linewidth=linewidth, color=palette['EI'])

# Uncomment to add shaded regions for std
# plt.fill_between(cost_NS_mean_TS1, coverage_NS_mean_TS1 - coverage_NS_std_TS1, coverage_NS_mean_TS1 + coverage_NS_std_TS1, alpha=alpha, color=palette['BEACON'])
# plt.fill_between(cost_MaxVar_mean, coverage_MaxVar_mean - coverage_MaxVar_std, coverage_MaxVar_mean + coverage_MaxVar_std, alpha=alpha, color=palette['MaxVar'])
# plt.fill_between(cost_GA_NS_novel_mean, coverage_GA_NS_novel_mean - coverage_GA_NS_novel_std, coverage_GA_NS_novel_mean + coverage_GA_NS_novel_std, alpha=alpha, color=palette['NS-EA'])
# plt.fill_between(cost_DEA_mean, coverage_DEA_mean - coverage_DEA_std, coverage_DEA_mean + coverage_DEA_std, alpha=alpha, color=palette['NS-DEA'])
# plt.fill_between(cost_NS_xspace_mean, coverage_NS_xspace_mean - coverage_NS_xspace_std, coverage_NS_xspace_mean + coverage_NS_xspace_std, alpha=alpha, color=palette['NS-EA-FS'])
# plt.fill_between(cost_sobol_mean, coverage_sobol_mean - coverage_sobol_std, coverage_sobol_mean + coverage_sobol_std, alpha=alpha, color=palette['Sobol'])
# plt.fill_between(cost_RS_mean, coverage_RS_mean - coverage_RS_std, coverage_RS_mean + coverage_RS_std, alpha=alpha, color=palette['RS'])
# plt.fill_between(cost_BO_mean, coverage_BO_mean - coverage_BO_std, coverage_BO_mean + coverage_BO_std, alpha=alpha, color=palette['EI'])

plt.ylim(0.48, 1.02)
plt.xlabel('Number of evaluations', fontsize=text_size, fontweight=weight)
plt.ylabel('Best Reward', fontsize=text_size, fontweight=weight)
plt.legend(prop={'weight':'bold','size':text_size}, loc=2)

plt.tick_params(axis='both', which='both', width=2)
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

# Violin plot
text_size = 18
plt.figure(figsize=(12,10))

# Convert tensors to lists and create a DataFrame
data = {name: tensor.tolist() for name, tensor in tensors.items()}
df = pd.DataFrame(data)

# Melt the DataFrame to long format
df_melted = df.melt(var_name='Tensor', value_name='Value')

# Create the violin plot with the same colors
sns.violinplot(x='Tensor', y='Value', data=df_melted, palette=palette, width=0.4, inner='point',cut=0)

plt.ylabel('Final Value', fontsize=text_size, fontweight='bold')
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



