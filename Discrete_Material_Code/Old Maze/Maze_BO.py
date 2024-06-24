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
from botorch import fit_gpytorch_model
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
sys.path.append('/home/tang.1856/Jonathan/Hands-on-Neuroevolution-with-Python-master/Chapter6')
from maze_NS import Maze
import visualize

def reachability_uniformity(behavior, n_bins = 25, obj_lb = -5, obj_ub = 5):
    behavior = behavior.squeeze(1).numpy()
    num = len(behavior)
    cum_hist, _ = np.histogram(behavior, np.linspace(obj_lb, obj_ub, n_bins + 1))
    cum_hist = cum_hist[np.nonzero(cum_hist)] / (num) # discrete distribution
    cum_hist_uni = np.mean(cum_hist) * np.ones_like(cum_hist) # theoretical uniform distribution

    cum_coverage = len(cum_hist) / n_bins
    cum_uniformity = 1 - jensenshannon(cum_hist, cum_hist_uni, base=2)
    
    return cum_coverage, cum_uniformity

def bo_run(X, y, r, nb_iterations, nb_COFs_initialization, which_acquisition):
    '''This function is based on the following code: https://github.com/gerbswastaken/BayesianOptimizationForMOFs/blob/317f63000bab4b5cddbb34e251a0b334b2f8e561/Python%20Notebooks/bayesian_optimization_hydrogen.ipynb'''
    assert nb_iterations > nb_COFs_initialization
    assert which_acquisition in ['EI','max sigma', 'RS']
    
    np.random.seed(r)
    ids_acquired = np.random.choice(np.arange((len(y))), size=5000, replace=False)
    X = X[ids_acquired]
    y = y[ids_acquired]
    # y = np.reshape(y, (np.size(y), 1)) # for the GP
       
    # X = torch.from_numpy(X)
    X = (X - X.min(dim=0).values)/(X.max(dim=0).values - X.min(dim=0).values) # normalize the original input data
    
    # select initial COFs for training data randomly.
    # idea is to keep populating this ids_acquired and return it for analysis.
    np.random.seed(r)
    ids_acquired = np.random.choice(np.arange((len(y))), size=nb_COFs_initialization, replace=False)
    
    X_unsqueezed = X.unsqueeze(1)
    # initialize acquired y, since it requires normalization
    y_acquired = y[ids_acquired]
    # standardize outputs using *only currently acquired data*
    y_acquired = (y_acquired - torch.mean(y_acquired)) / torch.std(y_acquired)
    
    coverage, uniformity = reachability_uniformity(y[ids_acquired], n_bins=n_bins, obj_lb = y.min(), obj_ub = y.max())
    coverage_list = [coverage]
    uniformity_list = [uniformity]
    cost_list = [0]
    cumbent_list = [float(max(y[ids_acquired]))]
    
    for i in range(nb_COFs_initialization, nb_iterations):
        
        # construct and fit GP model
        model = SingleTaskGP(X[ids_acquired, :].to(torch.float64), y_acquired.to(torch.float64))
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        fit_gpytorch_model(mll)
       
        # Set the lengthscale and outputscale
        # model.covar_module.base_kernel.lengthscale = 0.5 # Setting the lengthscale
        # model.covar_module.outputscale = 1.0  # Setting the outputscale 
        
        if which_acquisition == "max sigma":
            with torch.no_grad():
                acquisition_values = model.posterior(X_unsqueezed).variance.squeeze()
        elif which_acquisition == "EI":
            acquisition_function = ExpectedImprovement(model, best_f=y_acquired.max().item())
            with torch.no_grad(): # to avoid memory issues; we arent using the gradient...
               acquisition_values = acquisition_function.forward(X_unsqueezed) # runs out of memory
        elif which_acquisition == "RS":
            acquisition_values = torch.rand(len(X_unsqueezed))
        else:
            raise Exception("not a valid acquisition function")

        # select COF to acquire with maximal aquisition value, which is not in the acquired set already
        ids_sorted_by_aquisition = acquisition_values.argsort(descending=True)
        for id_max_aquisition_all in ids_sorted_by_aquisition:
            if not id_max_aquisition_all.item() in ids_acquired:
                id_max_aquisition = id_max_aquisition_all.item()
                break

        # acquire this COF
        ids_acquired = np.concatenate((ids_acquired, [id_max_aquisition]))
        assert np.size(ids_acquired) == i + 1
    
        # update y aquired; start over to normalize properly
        y_acquired = y[ids_acquired] # start over to normalize y properly
        y_acquired = (y_acquired - torch.mean(y_acquired)) / torch.std(y_acquired)
        
        cumbent_list.append(float(max(y[ids_acquired])))
        coverage, uniformity = reachability_uniformity(y[ids_acquired], n_bins=n_bins, obj_lb = y.min(), obj_ub = y.max())
        coverage_list.append(coverage)
        uniformity_list.append(uniformity)
        cost_list.append(cost_list[-1] + len([id_max_aquisition]))
        
        
    assert np.size(ids_acquired) == nb_iterations
    return ids_acquired, X[ids_acquired, :], y[ids_acquired], coverage_list, uniformity_list, cost_list, cumbent_list


def calculate_rmse(X, y, X_BO, Y_BO, ids_acquired):
    # ids_acquired_test = np.random.choice(np.arange((len(y))), size=len(y), replace=False) # sample all data from the original data set
    mask = np.ones(len(X),dtype=bool)
    mask[ids_acquired]=False
    X_test = X[mask]
    Y_test = y[mask].numpy()
    # Y_test = y[ids_acquired_test].numpy()
    # X_test = X[ids_acquired_test]
    
    model = SingleTaskGP(X_BO, Y_BO, outcome_transform=Standardize(m=1)) # X_BO and Y_BO are the sampled data for BO
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_model(mll)
    
    y_pred = model.posterior(X_test).mean.detach().numpy()
    rmse = np.sqrt(np.mean((Y_test - y_pred)**2))
    
    # Try plotting parity plot using the data we sampled 
    # plt.scatter(Y_test, y_pred)
    # # plt.plot([min(y), max(y)], [min(y), max(y)], 'k--', label='Ideal Fit') 
    # plt.xlabel('True value')
    # plt.ylabel('predicted value')
    # plt.title('Parity Plot (MaxVar)')
    # plt.grid(True)
    return rmse

if __name__ == '__main__':
    
    dim = 22
    n_bins = 25
    N_init = 10 # number of initial training data
    nb_iterations = 150 # iteration time for BO
    nb_runs = 20 # number of replicate 
    
    X = torch.load('X_original.pt')
    y = torch.load('y_original.pt')
    
    
    # y = torch.from_numpy(y)
    
    
    
    
    coverage_tensor = []
    uniformity_tensor = []
    cost_tensor = []
    rmse_list = []
    cumbent_tensor = []

    which_acquisition = "max sigma" # specify acquisition function
    nb_COFs_initializations = {"RS": [N_init],
                               "max sigma": [N_init],
                               "EI": [N_init]}
    
    
    for nb_COFs_initialization in nb_COFs_initializations[which_acquisition]: # different initial data point 
        print("# COFs in initialization:", nb_COFs_initialization)
        
        
        for r in range(nb_runs): # run different replicate 
            print("\nRUN", r)
            t0 = time.time()
            # Run the BO
            ids_acquired, X_BO, Y_BO, coverage_list, uniformity_list, cost_list, cumbent_list = bo_run(X, y, r, nb_iterations, nb_COFs_initialization, which_acquisition)
            
            coverage_tensor.append(coverage_list)
            uniformity_tensor.append(uniformity_list)
            cost_tensor.append(cost_list)
            cumbent_tensor.append(cumbent_list)
            # rmse = calculate_rmse(X, y, X_BO, Y_BO, ids_acquired) # use the sampled data to fit GP and calculate the rnse for unseen data
            # rmse_list.append(rmse)
        # print('Avg RMSE = ', sum(rmse_list)/len(rmse_list))
        
    
    ############################################################################################################################################################################
    # save the corresponding results
    cumbent_tensor = torch.tensor(cumbent_tensor, dtype = torch.float32)
    cost_tensor = torch.tensor(cost_tensor, dtype=torch.float32) 
    coverage_tensor = torch.tensor(coverage_tensor, dtype=torch.float32) 
    uniformity_tensor = torch.tensor(uniformity_tensor, dtype=torch.float32)  
    torch.save(coverage_tensor, 'MediumMaze_coverage_list_MaxVar.pt')
    torch.save(uniformity_tensor, 'MediumMaze_uniformity_list_MaxVar.pt')
    torch.save(cost_tensor, 'MediumMaze_cost_list_MaxVar.pt')  
    # torch.save(cost_tensor, 'H2_cumbent_list_MaxVar.pt') 
    #########################################################################################################################################################################
