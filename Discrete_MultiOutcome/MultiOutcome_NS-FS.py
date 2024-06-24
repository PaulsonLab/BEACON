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


def reachability_uniformity(behavior, n_bins = 25, mins = None, maxs=None, num_filled_grids = 100):
    
    filled_grids = set()
    grid_sizes = (maxs - mins) / (n_bins)
    # Assign points to grids
    for point in behavior:
        if all(point>=maxs):
            grid_indices = torch.tensor([n_bins-1,n_bins-1])
        else:
            grid_indices = ((point - mins) / grid_sizes).int()
        # Use a tuple (immutable) to store grid indices in the set
        filled_grids.add(tuple(grid_indices.tolist()))
        
        
    return len(filled_grids)/(num_filled_grids)
    
        
class CustomAcquisitionFunction():
    def __init__(self, sampled_X, k=10):
        
       
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
    
    dim = 25
    N_init = 50
    replicate = 20
    n_bins = 10
    BO_iter = 500
    k = 10
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
    y1 = data1['pure_uptake_CO2_298.00_15000']
    y2 = data1['pure_uptake_methane_298.00_580000']
    

    lb = torch.tensor(data1[feat].values.min(axis=0))
    ub = torch.tensor(data1[feat].values.max(axis=0))
    X_original = (torch.tensor(data1[feat].values) - lb)/(ub-lb)
    Y_original = torch.stack((torch.tensor(y1.values),torch.tensor(y2.values)),dim=1)
    
    obj_lb1 = min(y1.values)
    obj_ub1 = max(y1.values)
    obj_lb2 = min(y2.values)
    obj_ub2 = max(y2.values)
    
    mins = torch.min(Y_original, dim=0).values
    maxs = torch.max(Y_original, dim=0).values
    
    # Calculate the grid size for each dimension
    grid_sizes = (maxs - mins) / (n_bins)
    
    # Initialize an empty set to keep track of filled grids
    filled_grids = set()
    
    # Assign points to grids
    for point in Y_original:
        if all(point>=maxs):
            grid_indices = torch.tensor([n_bins-1,n_bins-1])
        else:
            grid_indices = ((point - mins) / grid_sizes).int()
        # Use a tuple (immutable) to store grid indices in the set
        filled_grids.add(tuple(grid_indices.tolist()))
    
    # Number of unique grids that contain at least one data point
    num_filled_grids = len(filled_grids)
    
    
    
    cost_tensor = []
    coverage_tensor = []
        
    
    for seed in range(replicate):
        print('seed:',seed)
        np.random.seed(seed)
        ids_acquired = np.random.choice(np.arange((len(Y_original))), size=N_init, replace=False)
        train_x = X_original[ids_acquired]
        train_y = Y_original[ids_acquired]
       
        coverage = reachability_uniformity(train_y, n_bins, mins,maxs, num_filled_grids)        
        coverage_list = [coverage]        
        cost_list = [0]
        
        
        # Start BO loop
        for i in range(BO_iter):        
            
            custom_acq_function = CustomAcquisitionFunction(train_x, k=k)
            
            # Optimize the acquisition function (discrete)
            acquisition_values = custom_acq_function(X_original)
            ids_sorted_by_aquisition = acquisition_values.argsort(descending=True)
            for id_max_aquisition_all in ids_sorted_by_aquisition:
                if not id_max_aquisition_all.item() in ids_acquired:
                    id_max_aquisition = id_max_aquisition_all.item()
                    break
                
            ids_acquired = np.concatenate((ids_acquired, [id_max_aquisition]))
            train_x = X_original[ids_acquired]
            train_y = Y_original[ids_acquired] # original scale
            
            coverage = reachability_uniformity(train_y, n_bins, mins,maxs, num_filled_grids)
            coverage_list.append(coverage)            
            cost_list.append(cost_list[-1] + len([id_max_aquisition]))
    
        
        cost_tensor.append(cost_list)
        coverage_tensor.append(coverage_list)
           
    cost_tensor = torch.tensor(cost_tensor, dtype=torch.float32) 
    coverage_tensor = torch.tensor(coverage_tensor, dtype=torch.float32) 
      
    torch.save(coverage_tensor, 'MOF_coverage_list_NS_xspace.pt') 
    torch.save(cost_tensor, 'MOF_cost_list_NS_xspace.pt')  

    
   
    


