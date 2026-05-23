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
import numpy as np
from botorch.models.transforms.outcome import Standardize
from scipy.spatial.distance import cdist, jensenshannon
import pandas as pd

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
       
        # construct and fit GP model
        model = SingleTaskGP(X[ids_acquired, :], y_acquired, outcome_transform=Standardize(m=1))
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        fit_gpytorch_mll(mll)
      
        if which_acquisition == "RS":
            acquisition_values = torch.rand(len(X_unsqueezed))
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
    
    n_bins = 25
    N_init = 10 
    BO_iterations = 100 
    replicate = 20 
    
    # Case Study 1: Hydrogen adsorption
    # load H2 adsorption MOF data from pkl file 
    #dim = 8
    # X = pickle.load(open('/fs/ess/PAS2983/jontwt/BEACON/Material Data/hydrogen_input_output.pkl', 'rb'))['x'] # data from Ghude and Chowdhury 2023 (7 features for MOFs)
    # y = pickle.load(open('/fs/ess/PAS2983/jontwt/BEACON/Material Data/hydrogen_input_output.pkl', 'rb'))['y'] # data from Ghude and Chowdhury 2023 (H2 adsorp capacity)
    
    # Case Study 2
    df = pd.read_csv('/fs/ess/PAS2983/jontwt/BEACON/Material Data/Nitrogen.csv') # data from Daglar et al.
    dim = 20
    X = (df.iloc[:, 1:(1+dim)]).values
    y = df['U_N2 (mol/kg)'].values
    
    # Case Study 3: Water Solubility
    # load water solubility data from csv file 
    # df = pd.read_csv('/fs/ess/PAS2983/jontwt/BEACON/Material Data/water_set_wide_descriptors.csv') # data from Qiao et al.
    # dim = 14
    # X = (df.iloc[:, 4:(4+dim)]).values
    # y = df.iloc[:, 3].values
    
    y = np.reshape(y, (np.size(y), 1)) # for the GP       
    X = torch.from_numpy(X)
    X = (X - X.min(dim=0).values)/(X.max(dim=0).values - X.min(dim=0).values) # normalize the original input data
    y = torch.from_numpy(y)
        
    coverage_tensor = []
    cost_tensor = []

    which_acquisition = "RS" # specify acquisition function
    nb_COFs_initializations = {"RS": [N_init]}
    
    
    for nb_COFs_initialization in nb_COFs_initializations[which_acquisition]: # different initial data point 
                
        for r in range(replicate): # run different replicate 
            
            # Run the BO
            ids_acquired, X_BO, Y_BO, coverage_list, cost_list = bo_run(X, y, r, BO_iterations, nb_COFs_initialization, which_acquisition)
            
            coverage_tensor.append(coverage_list)
            cost_tensor.append(cost_list)
               
    # save the corresponding results   
    cost_tensor = torch.tensor(cost_tensor, dtype=torch.float32) 
    coverage_tensor = torch.tensor(coverage_tensor, dtype=torch.float32) 
    # torch.save(coverage_tensor, 'N2uptake_coverage_list_EI.pt')
    # torch.save(cost_tensor, 'N2uptake_cost_list_EI.pt')  
   
  
