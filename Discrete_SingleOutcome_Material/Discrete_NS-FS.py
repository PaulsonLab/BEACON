#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 12 13:52:57 2024

@author: tang.1856
"""

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
from botorch.acquisition import AcquisitionFunction
from botorch.optim import optimize_acqf
from scipy.spatial.distance import cdist, jensenshannon
import numpy as np
from torch.quasirandom import SobolEngine
from botorch.test_functions import Rosenbrock, Ackley
import pickle
from botorch.models.transforms.outcome import Standardize
from botorch import fit_gpytorch_model
import matplotlib.pyplot as plt
import pandas as pd
import math


def reachability_uniformity(behavior, n_bins = 25, obj_lb = -5, obj_ub = 5, n_hist = 25):
    behavior = behavior.squeeze(1).numpy()
    num = len(behavior)
    cum_hist, _ = np.histogram(behavior, np.linspace(obj_lb, obj_ub, n_bins + 1))
    cum_hist = cum_hist[np.nonzero(cum_hist)] / (num) # discrete distribution
    cum_hist_uni = np.mean(cum_hist) * np.ones_like(cum_hist) # theoretical uniform distribution

    cum_coverage = len(cum_hist) / n_hist
    cum_uniformity = 1 - jensenshannon(cum_hist, cum_hist_uni, base=2)
    
    return cum_coverage, cum_uniformity
    
        
class CustomAcquisitionFunction():
    def __init__(self, model, sampled_X, k=10):
        
        self.model = model
        self.k = k
        self.sampled_X = sampled_X
        
    
    def __call__(self, X):
        """Compute the acquisition function value at X."""
        
        dist = torch.cdist(X.to(torch.float64), self.sampled_X.to(torch.float64))
        dist, _ = torch.sort(dist, dim = 1) # sort the distance 
        n = dist.size()[1]

        E = torch.cat((torch.ones(self.k), torch.zeros(n-self.k)), dim = 0) # find the k-nearest neighbor
        dist = dist*E
        acquisition_values = torch.sum(dist,dim=1)
        
        return acquisition_values.flatten()
       
if __name__ == '__main__':
    
    dim = 20
    N_init = 10
    replicate = 20
    n_bins = 25 
    k = 10    
    BO_iter = 500
    
    cost_tensor = []
    coverage_tensor = []
    
    # Case study 1
    # load H2 adsorption MOF data from pkl file 
    # X_original = pickle.load(open('/home/tang.1856/Jonathan/Novelty Search/Training Data/hydrogen_input_output.pkl', 'rb'))['x'] # data from Ghude and Chowdhury 2023 (7 features for MOFs)
    # y_original = pickle.load(open('/home/tang.1856/Jonathan/Novelty Search/Training Data/hydrogen_input_output.pkl', 'rb'))['y'] # data from Ghude and Chowdhury 2023 (H2 adsorp capacity)
    
    # Case study 2
    # load SourGas MOF data from xlsx file
    # df = pd.read_excel('/home/tang.1856/Jonathan/Novelty Search/Training Data/SourGas.xlsx') # data from Cho et al.
    # X_original = (df.iloc[:, 1:(1+dim)]).values
    # y_original = df.iloc[:, (1+dim)].values
    
    # Case study 3
    # load solubility data from csv file 
    # df = pd.read_csv('/home/tang.1856/Jonathan/Novelty Search/Training Data/water_set_wide_descriptors.csv') # data from Boobier et al.
    # X_original = (df.iloc[:, 4:(4+dim)]).values
    # y_original = df.iloc[:, 3].values
      
    # Case Study 4
    df = pd.read_csv('/home/tang.1856/Jonathan/Novelty Search/Training Data/rawdata/Nitrogen.csv') # data from Boobier et al.
    X_original = (df.iloc[:, 1:(1+dim)]).values
    y_original = df['U_N2 (mol/kg)'].values
    
    y_original = np.reshape(y_original, (np.size(y_original), 1)) 
    X_original = torch.from_numpy(X_original)
    
    X_original = (X_original - X_original.min(dim=0).values)/(X_original.max(dim=0).values - X_original.min(dim=0).values) 
    y_original = torch.from_numpy(y_original)
    
    n_hist = float(torch.count_nonzero(torch.histc(y_original,bins=n_bins)))
    
    obj_lb = y_original.min() # obj minimum
    obj_ub = y_original.max() # obj maximum 
    
    for seed in range(replicate):
        print('seed:',seed)
        np.random.seed(seed)
        ids_acquired = np.random.choice(np.arange((len(y_original))), size=N_init, replace=False)
        train_x = X_original[ids_acquired]
        train_y = y_original[ids_acquired] # original scale
       
        coverage, uniformity = reachability_uniformity(train_y, n_bins, obj_lb, obj_ub, n_hist) 
        
        coverage_list = [coverage]
        cost_list = [0] 
        
        # Start BO loop
        for i in range(BO_iter):        
            
            model = SingleTaskGP(train_x, train_y, outcome_transform=Standardize(m=1))
            mll = ExactMarginalLogLikelihood(model.likelihood, model)
            # fit_gpytorch_mll(mll)
            
            custom_acq_function = CustomAcquisitionFunction(model, train_x, k=k)
            
            # Optimize the acquisition function (discrete)
            acquisition_values = custom_acq_function(X_original)
            ids_sorted_by_aquisition = acquisition_values.argsort(descending=True)
            for id_max_aquisition_all in ids_sorted_by_aquisition:
                if not id_max_aquisition_all.item() in ids_acquired:
                    id_max_aquisition = id_max_aquisition_all.item()
                    break
                
            ids_acquired = np.concatenate((ids_acquired, [id_max_aquisition]))
            train_x = X_original[ids_acquired]
            train_y = y_original[ids_acquired] # original scale
            
            coverage, uniformity = reachability_uniformity(train_y, n_bins, obj_lb, obj_ub, n_hist)
            coverage_list.append(coverage)   
            cost_list.append(cost_list[-1] + len([id_max_aquisition]))
            
        cost_tensor.append(cost_list)
        coverage_tensor.append(coverage_list)
          
    cost_tensor = torch.tensor(cost_tensor, dtype=torch.float32) 
    coverage_tensor = torch.tensor(coverage_tensor, dtype=torch.float32)   
    torch.save(coverage_tensor, 'N2uptake_coverage_list_NS_xspace.pt')    
    torch.save(cost_tensor, 'N2uptake_cost_list_NS_xspace.pt')  
   
    


