#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  6 16:43:21 2024

@author: tang.1856
"""
import pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from gauche.dataloader import MolPropLoader,ReactionLoader
import os


# Case study 1
# X = pickle.load(open('/home/tang.1856/Jonathan/Novelty Search/Training Data/hydrogen_input_output.pkl', 'rb'))['x'] # data from Ghude and Chowdhury 2023 (7 features for MOFs)
# y = pickle.load(open('/home/tang.1856/Jonathan/Novelty Search/Training Data/hydrogen_input_output.pkl', 'rb'))['y'] # data from Ghude and Chowdhury 2023 (7 features for MOFs)

# Case Study 2
# df = pd.read_excel('/home/tang.1856/Jonathan/Novelty Search/Training Data/SourGas.xlsx') # data from Cho et al. 2020 (12 features for MOFs)
# # X = (df.iloc[:, 1:(1+dim)]).values
# y = df.iloc[:, (13)].values

# Case Study 3
# load solubility data from csv file 
# df = pd.read_csv('/home/tang.1856/Jonathan/Novelty Search/Training Data/water_set_wide_descriptors.csv') # data from Qiao et al.
# # X = (df.iloc[:, 4:(4+dim)]).values
# y = df.iloc[:, 3].values

# Case Study 4
y = pd.read_csv('/home/tang.1856/Jonathan/Novelty Search/Training Data/rawdata/Nitrogen.csv')['U_N2 (mol/kg)']

# Photoswitch
# Load the Photoswitch dataset
# loader = MolPropLoader()
# # # loader.load_benchmark("ESOL")
# # # loader = ReactionLoader()
# loader.load_benchmark("ESOL")
# # # loader.read_csv("/home/tang.1856/Jonathan/Novelty Search/MultiObj/SMILES_Malaria.csv_with_SAScore.csv", "SMILES","y1")
# # # We use the fragprints representations (a concatenation of Morgan fingerprints and RDKit fragment features)
# loader.featurize('ecfp_fragprints')
# # loader.featurize('drfp')
# # X_original = loader.features
# y = loader.labels.flatten()

#logD
# path = os.getcwd()
# data_path = "/home/tang.1856/Jonathan/Novelty Search/logP_logD/Training Data/extract_data_logD.csv"

# col_list=['LogD','Exp_RT']
# lc_df = pd.read_csv(data_path,usecols=col_list)

# # Remove non_retained molecules
# index=lc_df[lc_df['Exp_RT'] < 180].index
# lc_df.drop(lc_df[lc_df['Exp_RT'] < 180].index,inplace=True)

# # Import descriptor file
# # path = os.getcwd()
# data_path ="/home/tang.1856/Jonathan/Novelty Search/logP_logD/Training Data/descriptors_logD.csv"
# des_df = pd.read_csv(data_path,index_col=0)

# # Remove non_retained molecules
# des_df  = des_df.drop(des_df.index[index]) # modified from original code

# data_set_1 = des_df
# data_set_2 = lc_df
# des_with_lc = pd.concat([data_set_1,data_set_2],axis=1)
# des_with_lc_feat_corr = des_with_lc.columns[des_with_lc.corrwith(des_with_lc['LogD']) >=0.90][:-1]
# des_with_lc = des_with_lc.drop(columns=des_with_lc_feat_corr)

# # Filling the nan with mean values in des_with_lc
# for col in des_with_lc:
#     des_with_lc[col].fillna(des_with_lc[col].mean(),inplace=True)

# # Remove columns with zero vlues
# des_with_lc = des_with_lc.loc[:,(des_with_lc**2).sum() != 0]
# data = des_with_lc.drop(['LogD'],axis=1)

# # Remove features with low Variance(threshold<=0.05)
# data_var = data.var()
# del_feat = list(data_var[data_var <= 0.05].index)
# data.drop(columns=del_feat, inplace=True)

# # Remove features with correlation(threshold > 0.95)
# corr_matrix = data.corr().abs()
# mask = np.triu(np.ones_like(corr_matrix,dtype=bool))
# tri_df = corr_matrix.mask(mask)
# to_drop =  [ c for c in tri_df.columns if any(tri_df[c] > 0.95)]
# data = data.drop(to_drop,axis=1)
# X = (data.iloc[:, 1:(1+125)]).values
# y = des_with_lc['LogD'].values


weights = np.ones_like(y)/len(y)*100
# plt.hist(y, bins=25, weights=weights,alpha=0.75)

# Plotting
plt.figure(figsize=(8,6.5))
data = pd.DataFrame({'Y':y, 'Weights':weights})
sns.histplot(data=data, x='Y',bins=25, weights='Weights', kde=True ,color='green', alpha=0.3)
plt.ylabel('Frequency (%)', fontsize=18, fontweight='bold')
# plt.xlabel('H$_2$ uptake capacity')
plt.xlabel('Nitrogen uptake capacity', fontsize=18, fontweight='bold')
plt.tick_params(axis='both',
                which='both',
                width=2)
ax = plt.gca()          
for label in ax.get_xticklabels():
    label.set_fontsize(18)
    label.set_fontweight('bold')
    
for label in ax.get_yticklabels():
    label.set_fontsize(18)
    label.set_fontweight('bold')
    
ax.spines['top'].set_linewidth(2)
ax.spines['bottom'].set_linewidth(2)
ax.spines['left'].set_linewidth(2)
ax.spines['right'].set_linewidth(2)

#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Synthetic Multioutcome
import numpy as np
import matplotlib.pyplot as plt

def synthetic_function(x):
    y1 = np.sin(x[0]) * np.cos(x[1]) + x[2] * np.exp(-x[0]**2) * np.cos(x[0] + x[1]) + 0.01*np.sin((x[3] + x[4] + x[5]))
    y2 = np.sin(x[3]) * np.cos(x[4]) + x[5] * np.exp(-x[3]**2) * np.sin(x[3] + x[4]) + 0.01*np.cos((x[0] + x[1] + x[2]))
    # y2 = np.sin(x[0]) * np.cos(x[1]) + x[2] * np.exp(-x[0]**2) * np.sin(x[0] + x[1])
    return np.array([y1, y2])

# Generate random points in 6D
# np.random.seed(42)
inputs = np.random.uniform(-5, 5, (20000,6))  # Uniformly distributed in a range

# Calculate outcomes
outcomes = np.array([synthetic_function(x) for x in inputs])

# # MOF Multi-outcome
# import torch
# feat = set(
# ["func-chi-0-all" ,"D_func-S-3-all", "total_SA_volumetric", 
#  "Di", "Dif", "mc_CRY-Z-0-all","total_POV_volumetric","density [g/cm^3]", "total_SA_gravimetric",
#  "D_func-S-1-all","Df", "mc_CRY-S-0-all" ,"total_POV_gravimetric","D_func-alpha-1-all","func-S-0-all",
#  "D_mc_CRY-chi-3-all","D_mc_CRY-chi-1-all","func-alpha-0-all",
#  "D_mc_CRY-T-2-all","mc_CRY-Z-2-all","D_mc_CRY-chi-2-all",
# "total_SA_gravimetric","total_POV_gravimetric","Di","density [g/cm^3]",
#  "func-S-0-all",
#  "func-chi-2-all","func-alpha-0-all",
#  "total_POV_volumetric","D_func-alpha-1-all","total_SA_volumetric",
#  "func-alpha-1-all",
#  "func-alpha-3-all",
#  "Dif",
#  "Df",
#  "func-chi-3-all", 
#   'Di',
#  'Df',
#  'Dif',
#  'density [g/cm^3]',
#  'total_SA_volumetric',
#  'total_SA_gravimetric',
#  'total_POV_volumetric',
#  'total_POV_gravimetric'
# ])

# file_path1 = '/home/tang.1856/Downloads/PMOF20K_traindata_7000_train.csv'
# data1 = pd.read_csv(file_path1)
# y1 = data1['pure_uptake_CO2_298.00_15000']
# y2 = data1['pure_uptake_methane_298.00_580000']

# Y_original = torch.stack((torch.tensor(y1.values),torch.tensor(y2.values)),dim=1)
# # outcomes = Y_original.numpy()


# # file_path1 = '/home/tang.1856/Jonathan/Novelty Search/Training Data/rawdata/Nitrogen.csv'
# # data1 = pd.read_csv(file_path1)
# # y1 = data1['D_N2 (cm2/s)']
# # y2 = data1['U_N2 (mol/kg)']

# Y_original = torch.stack((torch.tensor(y1.values),torch.tensor(y2.values)),dim=1)
# outcomes = Y_original.numpy()

# # Plotting
plt.figure()
plt.scatter(outcomes[:, 0], outcomes[:, 1], alpha=0.5, edgecolors='none', s=4)
# plt.title('Visually Interesting 2D Outcome Space from High-Dimensional Inputs')
plt.tick_params(axis='both',
                which='both',
                width=2)
ax = plt.gca()          
for label in ax.get_xticklabels():
    label.set_fontsize(12)
    label.set_fontweight('bold')
    
for label in ax.get_yticklabels():
    label.set_fontsize(12)
    label.set_fontweight('bold')
    
ax.spines['top'].set_linewidth(2)
ax.spines['bottom'].set_linewidth(2)
ax.spines['left'].set_linewidth(2)
ax.spines['right'].set_linewidth(2)

# plt.xlabel(r'CO$_{\text{2}}$ uptake', fontsize=12, fontweight='bold')
# plt.ylabel(r'CH$_{\text{4}}$ uptake', fontsize=12, fontweight='bold')
# plt.xlabel(r'y$_\textbf{1}$', fontsize=12, fontweight='bold')
# plt.ylabel(r'y$_2$', fontsize=12, fontweight='bold')
plt.grid(True)
plt.show()



x = outcomes[:, 0]
y = outcomes[:, 1]



def scatter_hist(x, y, ax, ax_histx, ax_histy):
    # no labels
    ax_histx.tick_params(axis="x", labelbottom=False)
    ax_histy.tick_params(axis="y", labelleft=False)

    # the scatter plot:
    ax.scatter(x, y, alpha=0.5, edgecolors='none', s=4)
    
    # now determine nice limits by hand:
    binwidth = 0.25
    xymax = max(np.max(np.abs(x)), np.max(np.abs(y)))
    lim = (int(xymax/binwidth) + 1) * binwidth

    bins = np.arange(-lim, lim + binwidth, binwidth)
    ax_histx.hist(x, bins=bins, color='cornflowerblue', edgecolor='black', alpha=0.5)
    ax_histy.hist(y, bins=bins, orientation='horizontal', color='cornflowerblue', edgecolor='black', alpha=0.5)
    
    # custom_xlim = (-0.1, 8)  # Customize these values as needed
    # custom_ylim = (-0.1, 8)  # Customize these values as needed
    custom_xlim = (-5, 5)  # Customize these values as needed
    custom_ylim = (-5, 5)  # Customize these values as needed
    ax.set_xlim(custom_xlim)
    ax.set_ylim(custom_ylim)
    ax_histx.set_xlim(custom_xlim)
    ax_histy.set_ylim(custom_ylim)
    
    ax.set_xlabel('y1',fontweight='bold',fontsize=12)
    ax.set_ylabel('y2',fontweight='bold',fontsize=12)
    # for label in ax.get_xticklabels():
    #     label.set_fontsize(12)
    #     label.set_fontweight('bold')
        
    # for label in ax.get_yticklabels():
    #     label.set_fontsize(12)
    #     label.set_fontweight('bold')
        
    # ax.spines['top'].set_linewidth(2)
    # ax.spines['bottom'].set_linewidth(2)
    # ax.spines['left'].set_linewidth(2)
    # ax.spines['right'].set_linewidth(2)
    
    # Customize tick labels
    for axis in [ax, ax_histx, ax_histy]:
        for label in axis.get_xticklabels() + axis.get_yticklabels():
            label.set_fontsize(10)
            label.set_fontweight('bold')

    # Customize axis spines
    for axis in [ax, ax_histx, ax_histy]:
        for spine in axis.spines.values():
            spine.set_linewidth(1)
    
    
    
# Start with a square Figure.
fig = plt.figure(figsize=(6, 6))
# Add a gridspec with two rows and two columns and a ratio of 1 to 4 between
# the size of the marginal axes and the main axes in both directions.
# Also adjust the subplot parameters for a square plot.
gs = fig.add_gridspec(2, 2,  width_ratios=(4, 1), height_ratios=(1, 4),
                      left=0.1, right=0.9, bottom=0.1, top=0.9,
                      wspace=0.05, hspace=0.05)
# Create the Axes.
ax = fig.add_subplot(gs[1, 0])
ax_histx = fig.add_subplot(gs[0, 0], sharex=ax)
ax_histy = fig.add_subplot(gs[1, 1], sharey=ax)
# Draw the scatter plot and marginals.
scatter_hist(x, y, ax, ax_histx, ax_histy)
