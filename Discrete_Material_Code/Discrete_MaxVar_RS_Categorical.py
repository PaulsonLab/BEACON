#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 25 14:22:59 2024

@author: tang.1856
"""

import torch
from botorch.models import FixedNoiseGP, SingleTaskGP
from gpytorch.kernels import ScaleKernel
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch import fit_gpytorch_mll
from scipy.stats import norm
from botorch.acquisition.analytic import ExpectedImprovement
import matplotlib.pyplot as plt
import numpy as np
import pickle
import sys
import time
from botorch.models.transforms.outcome import Standardize
from scipy.spatial.distance import cdist, jensenshannon
import pandas as pd
from botorch.models.kernels.categorical import CategoricalKernel

def reachability_uniformity(behavior, n_bins = 25, obj_lb = -5, obj_ub = 5, n_hist = 25):
    behavior = behavior.squeeze(1).numpy()
    num = len(behavior)
    cum_hist, _ = np.histogram(behavior, np.linspace(obj_lb, obj_ub, n_bins + 1))
    cum_hist = cum_hist[np.nonzero(cum_hist)] / (num) # discrete distribution
    cum_hist_uni = np.mean(cum_hist) * np.ones_like(cum_hist) # theoretical uniform distribution

    cum_coverage = len(cum_hist) / n_hist
    cum_uniformity = 1 - jensenshannon(cum_hist, cum_hist_uni, base=2)
    
    return cum_coverage, cum_uniformity

def bo_run(X, y, r, BO_iterations, nb_COFs_initialization, which_acquisition):
    '''This function is based on the following code: https://github.com/gerbswastaken/BayesianOptimizationForMOFs/blob/317f63000bab4b5cddbb34e251a0b334b2f8e561/Python%20Notebooks/bayesian_optimization_hydrogen.ipynb'''
    assert BO_iterations > nb_COFs_initialization
    assert which_acquisition in ['max sigma', 'RS', 'EI']
    
    np.random.seed(r)
    ids_acquired = np.random.choice(np.arange((len(y))), size=nb_COFs_initialization, replace=False)
    
    X_unsqueezed = X.unsqueeze(1)
    y_acquired = y[ids_acquired]
    n_hist = float(torch.count_nonzero(torch.histc(y,bins=n_bins)))
    
    coverage, uniformity = reachability_uniformity(y[ids_acquired], n_bins=n_bins, obj_lb = y.min(), obj_ub = y.max(), n_hist=n_hist)
    coverage_list = [coverage]
    cost_list = [0]
    
    for i in range(BO_iterations):
        covar_module = CategoricalKernel(ard_num_dims = dim)
        # construct and fit GP model
        model = SingleTaskGP(X[ids_acquired, :], y_acquired, covar_module = covar_module, outcome_transform=Standardize(m=1))
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        # try:
        #     fit_gpytorch_mll(mll)
        # except:
        #     print('cant fit GP')
      
        if which_acquisition == "max sigma":
            with torch.no_grad():
                acquisition_values = model.posterior(X_unsqueezed).variance.squeeze()
        elif which_acquisition == "RS":
            acquisition_values = torch.rand(len(X_unsqueezed))
        elif which_acquisition == "EI":
            acquisition_function = ExpectedImprovement(model, best_f=y_acquired.max().item())
            with torch.no_grad():
                acquisition_values = acquisition_function(X_unsqueezed)
        else:
            raise Exception("not a valid acquisition function")

        
        ids_sorted_by_aquisition = acquisition_values.argsort(descending=True)
        for id_max_aquisition_all in ids_sorted_by_aquisition:
            if not id_max_aquisition_all.item() in ids_acquired:
                id_max_aquisition = id_max_aquisition_all.item()
                break

        ids_acquired = np.concatenate((ids_acquired, [id_max_aquisition]))   
        y_acquired = y[ids_acquired] 
        
        coverage, uniformity = reachability_uniformity(y[ids_acquired], n_bins=n_bins, obj_lb = y.min(), obj_ub = y.max(), n_hist=n_hist)
        coverage_list.append(coverage)
        cost_list.append(cost_list[-1] + len([id_max_aquisition]))
               
    return ids_acquired, X[ids_acquired, :], y[ids_acquired], coverage_list, cost_list

if __name__ == '__main__':
    
    dim = 8
    n_bins = 50
    N_init = 50 
    BO_iterations = 300 
    nb_runs = 5
    
    
        
    coverage_tensor = []
    cost_tensor = []

    which_acquisition = "EI" # specify acquisition function
    nb_COFs_initializations = {"RS": [N_init],
                               "max sigma": [N_init],
                               "EI": [N_init]}
    
    
    for nb_COFs_initialization in nb_COFs_initializations[which_acquisition]: # different initial data point 
                
        for r in range(nb_runs): # run different replicate 
            print("seed=", r)
            
            np.random.seed(0)
            # ids_acquired_original = np.random.choice(np.arange((65792)), size=10000, replace=False)
            indices = np.arange(65729)
            ids_acquired_original =np.random.choice(np.arange((65792)), size=10000, replace=False)
            X= np.load('/home/tang.1856/BEACON/tf_bind_8-SIX6_REF_R1/tf_bind_8-x-0.npy')[ids_acquired_original]
            y = np.load('/home/tang.1856/BEACON/tf_bind_8-SIX6_REF_R1/tf_bind_8-y-0.npy')[ids_acquired_original]
            
            # # Case Study 4
            # df = pd.read_csv('/home/tang.1856/Jonathan/Novelty Search/Training Data/rawdata/Nitrogen.csv') # data from Boobier et al.
            # X = (df.iloc[:, 1:(1+dim)]).values
            # y = df['U_N2 (mol/kg)'].values
            
            y = np.reshape(y, (np.size(y), 1)) # for the GP       
            X = torch.from_numpy(X).to(torch.float64)
            X = (X - X.min(dim=0).values)/(X.max(dim=0).values - X.min(dim=0).values) # normalize the original input data
            y = torch.from_numpy(y).to(torch.float64)
            # Run the BO
            ids_acquired, X_BO, Y_BO, coverage_list, cost_list = bo_run(X, y, r, BO_iterations, nb_COFs_initialization, which_acquisition)
            
            coverage_tensor.append(coverage_list)
            cost_tensor.append(cost_list)
               
    # save the corresponding results   
    cost_tensor = torch.tensor(cost_tensor, dtype=torch.float32) 
    coverage_tensor = torch.tensor(coverage_tensor, dtype=torch.float32) 
    torch.save(coverage_tensor, 'DNA_coverage_list_EI.pt')
    torch.save(cost_tensor, 'DNA_cost_list_EI.pt')  
   
  
