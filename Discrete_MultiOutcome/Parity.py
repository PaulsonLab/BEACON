#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May  5 14:45:56 2024

@author: tang.1856
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import gymnasium as gym
import torch
import gpytorch
from botorch.models import SingleTaskGP
from gpytorch.kernels import MaternKernel, RFFKernel, ScaleKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.mlls import ExactMarginalLogLikelihood
from torch.quasirandom import SobolEngine
from botorch.fit import fit_gpytorch_mll
from botorch.utils import standardize
import botorch
from typing import Tuple
from botorch.utils.transforms import t_batch_mode_transform
from botorch.acquisition import AcquisitionFunction, AnalyticAcquisitionFunction
from botorch.optim import optimize_acqf
from scipy.spatial.distance import cdist, jensenshannon
import numpy as np
from torch.quasirandom import SobolEngine
from botorch.test_functions import Rosenbrock, Ackley, Hartmann, StyblinskiTang
import torchsort
import pickle
from botorch.models.transforms.outcome import Standardize
import matplotlib.pyplot as plt
from gpytorch.kernels import RBFKernel, ScaleKernel

# file_path = '/home/tang.1856/Downloads/AllProps_1400BzNSN.csv'

# data = pd.read_csv(file_path)

# ered = data['Ered']

# Gsolv = data['Gsol']

# Abs = data['Absorption Wavelength']

# ids_acquired = np.random.choice(np.arange((len(ered))), size=100, replace=False)
# # plt.scatter(Abs[ids_acquired], Gsolv[ids_acquired])
# # plt.scatter(Gsolv[ids_acquired], ered[ids_acquired])
# plt.scatter(ered[ids_acquired], Abs[ids_acquired])
# fig = plt.figure()

# # # Add a 3D subplot
# ax = fig.add_subplot(111, projection='3d')

# # Scatter plot
# ax.scatter(ered, Gsolv, Abs)


# file_path = '/home/tang.1856/Downloads/removed_duplicates_of_150Kdataset.csv'
# data = pd.read_csv(file_path)

# ered = data['redox_potential']

# Gsolv = data['solvation_free_energy']
# ids_acquired = np.random.choice(np.arange((len(ered))), size=50000, replace=False)
# plt.scatter(Gsolv[ids_acquired], ered[ids_acquired])

# file_path1 = '/home/tang.1856/Downloads/b1-b21_random_deltaG.csv'
# file_path2 = '/home/tang.1856/Downloads/rg_results.csv'
# file_path3 = '/home/tang.1856/Downloads/b1-b21_random_virial_large_new.csv'

# data1 = pd.read_csv(file_path1)
# data2 = pd.read_csv(file_path2)
# data3 = pd.read_csv(file_path3)
# G = data1['deltaGmin']

# Rg = data2['Rg']

# Gmax =   data3['deltaGmax']

# Abs = data['Absorption Wavelength']

# ids_acquired = np.random.choice(np.arange((len(G))), size=len(G), replace=False)
# # plt.scatter(Abs[ids_acquired], Gsolv[ids_acquired])
# # plt.scatter(Gsolv[ids_acquired], ered[ids_acquired])
# plt.scatter(Rg[ids_acquired], Gmax[ids_acquired])

dim = 25
feat = set(
["func-chi-0-all" ,"D_func-S-3-all", "total_SA_volumetric", 
 "Di", "Dif", "mc_CRY-Z-0-all","total_POV_volumetric","density [g/cm^3]", "total_SA_gravimetric",
 "D_func-S-1-all","Df", "mc_CRY-S-0-all" ,"total_POV_gravimetric","D_func-alpha-1-all","func-S-0-all",
 "D_mc_CRY-chi-3-all","D_mc_CRY-chi-1-all","func-alpha-0-all",
 "D_mc_CRY-T-2-all","mc_CRY-Z-2-all","D_mc_CRY-chi-2-all",
"total_SA_gravimetric","total_POV_gravimetric","Di","density [g/cm^3]",
 "func-S-0-all",
 "func-chi-2-all","func-alpha-0-all",
 "total_POV_volumetric","D_func-alpha-1-all","total_SA_volumetric",
 "func-alpha-1-all",
 "func-alpha-3-all",
 "Dif",
 "Df",
 "func-chi-3-all", 
  'Di',
 'Df',
 'Dif',
 'density [g/cm^3]',
 'total_SA_volumetric',
 'total_SA_gravimetric',
 'total_POV_volumetric',
 'total_POV_gravimetric'
])

file_path1 = '/home/tang.1856/Downloads/PMOF20K_traindata_7000_train.csv'
data1 = pd.read_csv(file_path1)
y = data1['pure_uptake_CO2_298.00_15000']
y2 = data1['pure_uptake_methane_298.00_580000']
ids_acquired = np.random.choice(np.arange((len(y))), size=50, replace=False)

lb = torch.tensor(data1[feat].values.min(axis=0))
ub = torch.tensor(data1[feat].values.max(axis=0))
X_original = (torch.tensor(data1[feat].values) - lb)/(ub-lb)
Y_original = torch.tensor(y.values)
train_x = X_original[ids_acquired]
train_y = (Y_original[ids_acquired]).unsqueeze(1)

# plt.scatter(y[ids_acquired],y2[ids_acquired], s=0.2)

# covar_module = ScaleKernel(RBFKernel(ard_num_dims=dim))
model = SingleTaskGP(train_x.to(torch.float64), train_y, outcome_transform=Standardize(m=1))
mll = ExactMarginalLogLikelihood(model.likelihood, model)
fit_gpytorch_mll(mll)

ids_acquired = np.random.choice(np.arange((len(y))), size=5000, replace=False)
test_x = X_original[ids_acquired]
test_y = Y_original[ids_acquired]
posterior = model.posterior(X=test_x)
pred = posterior.mean
plt.scatter(test_y.detach().numpy(), pred.detach().numpy(), s=0.2)
