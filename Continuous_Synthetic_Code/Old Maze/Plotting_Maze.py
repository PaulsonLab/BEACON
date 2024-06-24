#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 16 14:38:55 2024

@author: tang.1856
"""
import torch
import matplotlib.pyplot as plt

synthetic = 'MediumMaze600'

cost_NS_TS1 = torch.load('/home/tang.1856/Jonathan/Novelty Search/Continuous_Synthetic_Code/Results/'+synthetic+'/'+synthetic+'_TS_1_cost_list_NS.pt')
# uniformity_NS_TS1 = torch.load('/home/tang.1856/Jonathan/Novelty Search/Continuous_Synthetic_Code/Results/'+synthetic+'/'+synthetic+'_TS_1_uniformity_list_NS.pt')
# coverage_NS_TS1 = torch.load('/home/tang.1856/Jonathan/Novelty Search/Continuous_Synthetic_Code/Results/'+synthetic+'/'+synthetic+'_TS_1_coverage_list_NS.pt')
cumbent_NS_TS1 = torch.load('/home/tang.1856/Jonathan/Novelty Search/Continuous_Synthetic_Code/Results/'+synthetic+'/'+synthetic+'_TS_1_cumbent_list_NS.pt')

cost_EI = torch.load('/home/tang.1856/Jonathan/Novelty Search/Continuous_Synthetic_Code/Results/'+synthetic+'/'+synthetic+'_cost_list_EI.pt')
# uniformity_EI = torch.load('/home/tang.1856/Jonathan/Novelty Search/Continuous_Synthetic_Code/Results/'+synthetic+'/'+synthetic+'_uniformity_list_EI.pt')
# coverage_EI = torch.load('/home/tang.1856/Jonathan/Novelty Search/Continuous_Synthetic_Code/Results/'+synthetic+'/'+synthetic+'_coverage_list_EI.pt')
cumbent_EI = torch.load('/home/tang.1856/Jonathan/Novelty Search/Continuous_Synthetic_Code/Results/'+synthetic+'/'+synthetic+'_cumbent_list_EI.pt')


cost_MaxVar = torch.load('/home/tang.1856/Jonathan/Novelty Search/Continuous_Synthetic_Code/Results/'+synthetic+'/'+synthetic+'_cost_list_MaxVar.pt')
# # uniformity_MaxVar = torch.load('/home/tang.1856/Jonathan/Novelty Search/Continuous_Synthetic_Code/Results/'+synthetic+'/'+synthetic+'_uniformity_list_MaxVar.pt')
# # coverage_MaxVar = torch.load('/home/tang.1856/Jonathan/Novelty Search/Continuous_Synthetic_Code/Results/'+synthetic+'/'+synthetic+'_coverage_list_MaxVar.pt')
cumbent_MaxVar = torch.load('/home/tang.1856/Jonathan/Novelty Search/Continuous_Synthetic_Code/Results/'+synthetic+'/'+synthetic+'_cumbent_list_MaxVar.pt')

cost_RS = torch.load('/home/tang.1856/Jonathan/Novelty Search/Continuous_Synthetic_Code/Results/'+synthetic+'/'+synthetic+'_cost_list_RS.pt')
# uniformity_RS = torch.load('/home/tang.1856/Jonathan/Novelty Search/Continuous_Synthetic_Code/Results/'+synthetic+'/'+synthetic+'_uniformity_list_RS.pt')
# coverage_RS = torch.load('/home/tang.1856/Jonathan/Novelty Search/Continuous_Synthetic_Code/Results/'+synthetic+'/'+synthetic+'_coverage_list_RS.pt')
cumbent_RS = torch.load('/home/tang.1856/Jonathan/Novelty Search/Continuous_Synthetic_Code/Results/'+synthetic+'/'+synthetic+'_cumbent_list_RS.pt')


# cost_GA_NS_random = torch.load('/home/tang.1856/Jonathan/Novelty Search/Continuous_Synthetic_Code/Results/'+synthetic+'/'+synthetic+'_cost_list_GA_random.pt')
# # uniformity_GA_NS_random = torch.load('/home/tang.1856/Jonathan/Novelty Search/Continuous_Synthetic_Code/Results/'+synthetic+'/'+synthetic+'_uniformity_list_GA_random.pt')
# # coverage_GA_NS_random = torch.load('/home/tang.1856/Jonathan/Novelty Search/Continuous_Synthetic_Code/Results/'+synthetic+'/'+synthetic+'_coverage_list_GA_random.pt')
# cumbent_GA_NS_random = torch.load('/home/tang.1856/Jonathan/Novelty Search/Continuous_Synthetic_Code/Results/'+synthetic+'/'+synthetic+'_cumbent_list_GA_random.pt')

cost_GA_NS_novel = torch.load('/home/tang.1856/Jonathan/Novelty Search/Continuous_Synthetic_Code/Results/'+synthetic+'/'+synthetic+'_cost_list_GA_novelty.pt')
# uniformity_GA_NS_novel = torch.load('/home/tang.1856/Jonathan/Novelty Search/Continuous_Synthetic_Code/Results/'+synthetic+'/'+synthetic+'_uniformity_list_GA_novelty.pt')
# coverage_GA_NS_novel = torch.load('/home/tang.1856/Jonathan/Novelty Search/Continuous_Synthetic_Code/Results/'+synthetic+'/'+synthetic+'_coverage_list_GA_novelty.pt')
cumbent_GA_NS_novel = torch.load('/home/tang.1856/Jonathan/Novelty Search/Continuous_Synthetic_Code/Results/'+synthetic+'/'+synthetic+'_cumbent_list_GA_novelty.pt')

cost_DEA = torch.load('/home/tang.1856/Jonathan/Novelty Search/Continuous_Synthetic_Code/Results/'+synthetic+'/'+synthetic+'_cost_list_GA_DEA.pt')
# # uniformity_DEA = torch.load('/home/tang.1856/Jonathan/Novelty Search/Continuous_Synthetic_Code/Results/'+synthetic+'/'+synthetic+'_uniformity_list_GA_DEA.pt')
# # coverage_DEA = torch.load('/home/tang.1856/Jonathan/Novelty Search/Continuous_Synthetic_Code/Results/'+synthetic+'/'+synthetic+'_coverage_list_GA_DEA.pt')
cumbent_DEA = torch.load('/home/tang.1856/Jonathan/Novelty Search/Continuous_Synthetic_Code/Results/'+synthetic+'/'+synthetic+'_cumbent_list_GA_DEA.pt')

cost_sobol = torch.load('/home/tang.1856/Jonathan/Novelty Search/Continuous_Synthetic_Code/Results/'+synthetic+'/'+synthetic+'_cost_list_sobol.pt')
# # uniformity_sobol = torch.load('/home/tang.1856/Jonathan/Novelty Search/Continuous_Synthetic_Code/Results/'+synthetic+'/'+synthetic+'_uniformity_list_sobol.pt')
# # coverage_sobol = torch.load('/home/tang.1856/Jonathan/Novelty Search/Continuous_Synthetic_Code/Results/'+synthetic+'/'+synthetic+'_coverage_list_sobol.pt')
cumbent_sobol = torch.load('/home/tang.1856/Jonathan/Novelty Search/Continuous_Synthetic_Code/Results/'+synthetic+'/'+synthetic+'_cumbent_list_sobol.pt')

cost_NS_xspace = torch.load('/home/tang.1856/Jonathan/Novelty Search/Continuous_Synthetic_Code/Results/'+synthetic+'/'+synthetic+'_cost_list_NS_x_space.pt')
# # uniformity_NS_xspace = torch.load('/home/tang.1856/Jonathan/Novelty Search/Continuous_Synthetic_Code/Results/'+synthetic+'/'+synthetic+'_uniformity_list_NS_x_space.pt')
# # coverage_NS_xspace = torch.load('/home/tang.1856/Jonathan/Novelty Search/Continuous_Synthetic_Code/Results/'+synthetic+'/'+synthetic+'_coverage_list_NS_x_space.pt')
cumbent_NS_xspace = torch.load('/home/tang.1856/Jonathan/Novelty Search/Continuous_Synthetic_Code/Results/'+synthetic+'/'+synthetic+'_cumbent_list_NS_x_space.pt')

# cost_SAAS = torch.load('Results/WaterSolubility_cost_list_NS_SAAS.pt')
# uniformity_SAAS = torch.load('Results/WaterSolubility_uniformity_list_NS_SAAS.pt')
# coverage_SAAS = torch.load('Results/WaterSolubility_coverage_list_NS_SAAS.pt')

cumbent_EI_mean = torch.mean(cumbent_EI, dim = 0)
cumbent_EI_std = torch.std(cumbent_EI, dim = 0)
# coverage_EI_mean = torch.mean(coverage_EI, dim = 0)
# coverage_EI_std = torch.std(coverage_EI, dim = 0)
# uniformity_EI_mean = torch.mean(uniformity_EI, dim = 0)
# uniformity_EI_std = torch.std(uniformity_EI, dim = 0)
cost_EI_mean = torch.mean(cost_EI, dim = 0)

cumbent_NS_mean_TS1 = torch.mean(cumbent_NS_TS1, dim = 0)
cumbent_NS_std_TS1 = torch.std(cumbent_NS_TS1, dim = 0)
# coverage_NS_mean_TS1 = torch.mean(coverage_NS_TS1, dim = 0)
# coverage_NS_std_TS1 = torch.std(coverage_NS_TS1, dim = 0)
# uniformity_NS_mean_TS1 = torch.mean(uniformity_NS_TS1, dim = 0)
# uniformity_NS_std_TS1 = torch.std(uniformity_NS_TS1, dim = 0)
cost_NS_mean_TS1 = torch.mean(cost_NS_TS1, dim = 0)

cumbent_MaxVar_mean = torch.mean(cumbent_MaxVar, dim = 0)
cumbent_MaxVar_std = torch.std(cumbent_MaxVar, dim = 0)
# # coverage_MaxVar_mean = torch.mean(coverage_MaxVar, dim = 0)
# # coverage_MaxVar_std = torch.std(coverage_MaxVar, dim = 0)
# # uniformity_MaxVar_mean = torch.mean(uniformity_MaxVar, dim = 0)
# # uniformity_MaxVar_std = torch.std(uniformity_MaxVar, dim = 0)
cost_MaxVar_mean = torch.mean(cost_MaxVar, dim = 0)

cumbent_RS_mean = torch.mean(cumbent_RS, dim = 0)
cumbent_RS_std = torch.std(cumbent_RS, dim = 0)
# coverage_RS_mean = torch.mean(coverage_RS, dim = 0)
# coverage_RS_std = torch.std(coverage_RS, dim = 0)
# uniformity_RS_mean = torch.mean(uniformity_RS, dim = 0)
# uniformity_RS_std = torch.std(uniformity_RS, dim = 0)
cost_RS_mean = torch.mean(cost_RS, dim = 0)

# coverage_NS_mean_mean = torch.mean(coverage_NS_mean, dim = 0)
# coverage_NS_mean_std = torch.std(coverage_NS_mean, dim = 0)
# uniformity_NS_mean_mean = torch.mean(uniformity_NS_mean, dim = 0)
# uniformity_NS_mean_std = torch.std(uniformity_NS_mean, dim = 0)
# cost_NS_mean_mean = torch.mean(cost_NS_mean, dim = 0)

# coverage_GA_NS_random_mean = torch.mean(coverage_GA_NS_random , dim = 0)
# coverage_GA_NS_random_std = torch.std(coverage_GA_NS_random , dim = 0)
# cumbent_GA_NS_random_mean = torch.mean(cumbent_GA_NS_random , dim = 0)
# cumbent_GA_NS_random_std = torch.std(cumbent_GA_NS_random , dim = 0)
# # uniformity_GA_NS_random_mean = torch.mean(uniformity_GA_NS_random , dim = 0)
# # uniformity_GA_NS_random_std = torch.std(uniformity_GA_NS_random , dim = 0)
# cost_GA_NS_random_mean = torch.mean(cost_GA_NS_random , dim = 0)

# coverage_GA_NS_novel_mean = torch.mean(coverage_GA_NS_novel, dim = 0)
# coverage_GA_NS_novel_std = torch.std(coverage_GA_NS_novel , dim = 0)
cumbent_GA_NS_novel_mean = torch.mean(cumbent_GA_NS_novel, dim = 0)
cumbent_GA_NS_novel_std = torch.std(cumbent_GA_NS_novel , dim = 0)
# uniformity_GA_NS_novel_mean = torch.mean(uniformity_GA_NS_novel, dim = 0)
# uniformity_GA_NS_novel_std = torch.std(uniformity_GA_NS_novel , dim = 0)
cost_GA_NS_novel_mean = torch.mean(cost_GA_NS_novel , dim = 0)

# # coverage_DEA_mean = torch.mean(coverage_DEA, dim = 0)
# # coverage_DEA_std = torch.std(coverage_DEA , dim = 0)
cumbent_DEA_mean = torch.mean(cumbent_DEA, dim = 0)
cumbent_DEA_std = torch.std(cumbent_DEA , dim = 0)
# # uniformity_DEA_mean = torch.mean(uniformity_DEA, dim = 0)
# # uniformity_DEA_std = torch.std(uniformity_DEA , dim = 0)
cost_DEA_mean = torch.mean(cost_DEA , dim = 0)

# # coverage_sobol_mean = torch.mean(coverage_sobol, dim = 0)
# # coverage_sobol_std = torch.std(coverage_sobol , dim = 0)
cumbent_sobol_mean = torch.mean(cumbent_sobol, dim = 0)
cumbent_sobol_std = torch.std(cumbent_sobol , dim = 0)
# # uniformity_sobol_mean = torch.mean(uniformity_sobol, dim = 0)
# # uniformity_sobol_std = torch.std(uniformity_sobol , dim = 0)
cost_sobol_mean = torch.mean(cost_sobol , dim = 0)

# # coverage_NS_xspace_mean = torch.mean(coverage_NS_xspace, dim = 0)
# # coverage_NS_xspace_std = torch.std(coverage_NS_xspace , dim = 0)
cumbent_NS_xspace_mean = torch.mean(cumbent_NS_xspace, dim = 0)
cumbent_NS_xspace_std = torch.std(cumbent_NS_xspace , dim = 0)
# # uniformity_NS_xspace_mean = torch.mean(uniformity_NS_xspace, dim = 0)
# # uniformity_NS_xspace_std = torch.std(uniformity_NS_xspace , dim = 0)
cost_NS_xspace_mean = torch.mean(cost_NS_xspace , dim = 0)



marker_interval = 7
plt.figure()
plt.plot(cost_MaxVar_mean[::marker_interval], cumbent_MaxVar_mean[::marker_interval], label='MaxVar', marker='^', markersize=6)
plt.plot(cost_NS_mean_TS1[::marker_interval], cumbent_NS_mean_TS1[::marker_interval], label='NS-BO(TS=1)', marker='x', markersize=6)

# plt.plot(cost_NS_mean_TS5, coverage_NS_mean_TS5, label='NS-BO(TS=5)')
plt.plot(cost_RS_mean[::marker_interval], cumbent_RS_mean[::marker_interval], label='RS', marker='v', markersize=6)
# plt.plot(cost_NS_mean_mean[::marker_interval], coverage_NS_mean_mean[::marker_interval], label='NS-BO(mean)', marker='*', markersize=6)
# plt.plot(cost_GA_NS_random_mean, cumbent_GA_NS_random_mean, label='NS-EA-Random', marker='s', markersize=6)
plt.plot(cost_GA_NS_novel_mean, cumbent_GA_NS_novel_mean, label='NS-EA-Novelty', marker='>', markersize=6)
plt.plot(cost_DEA_mean, cumbent_DEA_mean, label='DEA', marker='D', markersize=6)
plt.plot(cost_sobol_mean[::marker_interval], cumbent_sobol_mean[::marker_interval], label='sobol', marker='o', markersize=6)
plt.plot(cost_NS_xspace_mean[::marker_interval], cumbent_NS_xspace_mean[::marker_interval], label='NS-xspace', marker='p', markersize=6)
plt.plot(cost_EI_mean[::marker_interval], cumbent_EI_mean[::marker_interval], label='EI', marker='o', markersize=6)

# plt.plot(cost_SAAS_mean, coverage_SAAS_mean, label='NS-SAAS')
plt.fill_between(cost_MaxVar_mean, cumbent_MaxVar_mean - cumbent_MaxVar_std, cumbent_MaxVar_mean + cumbent_MaxVar_std,  alpha=0.3)
plt.fill_between(cost_NS_mean_TS1, cumbent_NS_mean_TS1 - cumbent_NS_std_TS1, cumbent_NS_mean_TS1 + cumbent_NS_std_TS1,  alpha=0.3)
# plt.fill_between(cost_NS_mean_TS5, coverage_NS_mean_TS5 - coverage_NS_std_TS5, coverage_NS_mean_TS5 + coverage_NS_std_TS5,  alpha=0.5)
plt.fill_between(cost_RS_mean, cumbent_RS_mean - cumbent_RS_std, cumbent_RS_mean + cumbent_RS_std,  alpha=0.3)

# plt.fill_between(cost_NS_mean_mean, coverage_NS_mean_mean - coverage_NS_mean_std, coverage_NS_mean_mean + coverage_NS_mean_std,  alpha=0.3)
# plt.fill_between(cost_GA_NS_random_mean, cumbent_GA_NS_random_mean - cumbent_GA_NS_random_std, cumbent_GA_NS_random_mean + cumbent_GA_NS_random_std,  alpha=0.3)
plt.fill_between(cost_GA_NS_novel_mean, cumbent_GA_NS_novel_mean - cumbent_GA_NS_novel_std, cumbent_GA_NS_novel_mean + cumbent_GA_NS_novel_std,  alpha=0.3)
plt.fill_between(cost_DEA_mean, cumbent_DEA_mean - cumbent_DEA_std, cumbent_DEA_mean + cumbent_DEA_std,  alpha=0.3)
plt.fill_between(cost_sobol_mean, cumbent_sobol_mean - cumbent_sobol_std, cumbent_sobol_mean + cumbent_sobol_std,  alpha=0.3)
plt.fill_between(cost_NS_xspace_mean, cumbent_NS_xspace_mean - cumbent_NS_xspace_std, cumbent_NS_xspace_mean + cumbent_NS_xspace_std,  alpha=0.3)
# plt.fill_between(cost_NS_softsort_mean, coverage_NS_softsort_mean - coverage_NS_softsort_std, coverage_NS_softsort_mean + coverage_NS_softsort_std,  alpha=0.3)
# plt.fill_between(cost_SAAS_mean, coverage_SAAS_mean - coverage_SAAS_std, coverage_SAAS_mean + coverage_SAAS_std,  alpha=0.5)
plt.fill_between(cost_EI_mean, cumbent_EI_mean - cumbent_EI_std, cumbent_EI_mean + cumbent_EI_std,  alpha=0.3)
plt.xlabel('Sampled data')
plt.ylabel('Fitness Value')
plt.legend()
plt.title(synthetic+'(d=22)')

# plt.figure()
# plt.plot(cost_BO_mean, uniformity_BO_mean, label='MaxVar')
# plt.plot(cost_NS_mean_TS1, uniformity_NS_mean_TS1, label='NS-BO(TS=1)')
# # plt.plot(cost_NS_mean_TS5, uniformity_NS_mean_TS5, label='NS-BO(TS=5)')
# plt.plot(cost_RS_mean, uniformity_RS_mean, label='RS')
# plt.plot(cost_NS_mean_mean, uniformity_NS_mean_mean, label='NS-BO(mean)')
# plt.plot(cost_GA_NS_random_mean, uniformity_GA_NS_random_mean, label='NS-EA-RS')
# plt.plot(cost_GA_NS_novel_mean, uniformity_GA_NS_novel_mean, label='NS-EA-novelty')
# plt.plot(cost_DEA_mean, uniformity_DEA_mean, label='DEA')
# # plt.plot(cost_SAAS_mean, uniformity_SAAS_mean, label='NS-SAAS')
# plt.fill_between(cost_BO_mean, uniformity_BO_mean - uniformity_BO_std, uniformity_BO_mean + uniformity_BO_std,  alpha=0.5)
# plt.fill_between(cost_NS_mean_TS1, uniformity_NS_mean_TS1 - uniformity_NS_std_TS1, uniformity_NS_mean_TS1 + uniformity_NS_std_TS1,  alpha=0.5)
# # plt.fill_between(cost_NS_mean_TS5, uniformity_NS_mean_TS5 - uniformity_NS_std_TS5, uniformity_NS_mean_TS5 + uniformity_NS_std_TS5,  alpha=0.5)
# plt.fill_between(cost_RS_mean, uniformity_RS_mean - uniformity_RS_std, uniformity_RS_mean + uniformity_RS_std,  alpha=0.5)
# plt.fill_between(cost_NS_mean_mean, uniformity_NS_mean_mean - uniformity_NS_mean_std, uniformity_NS_mean_mean + uniformity_NS_mean_std,  alpha=0.5)
# plt.fill_between(cost_GA_NS_random_mean, uniformity_GA_NS_random_mean - uniformity_GA_NS_random_std, uniformity_GA_NS_random_mean + uniformity_GA_NS_random_std,  alpha=0.5)
# plt.fill_between(cost_GA_NS_novel_mean, uniformity_GA_NS_novel_mean - uniformity_GA_NS_novel_std, uniformity_GA_NS_novel_mean + uniformity_GA_NS_novel_std,  alpha=0.5)
# plt.fill_between(cost_DEA_mean, uniformity_DEA_mean - uniformity_DEA_std, uniformity_DEA_mean + uniformity_DEA_std,  alpha=0.5)
# # plt.fill_between(cost_SAAS_mean, uniformity_SAAS_mean - uniformity_SAAS_std, uniformity_SAAS_mean + uniformity_SAAS_std,  alpha=0.5)
# plt.xlabel('Sampled data')
# plt.ylabel('Uniformity')
# plt.legend()

# plt.title(synthetic)