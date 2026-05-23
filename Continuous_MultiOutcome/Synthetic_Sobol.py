#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 11 13:25:40 2024

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
from botorch.acquisition import AcquisitionFunction, AnalyticAcquisitionFunction
from botorch.optim import optimize_acqf
from scipy.spatial.distance import cdist, jensenshannon
import numpy as np
from torch.quasirandom import SobolEngine
from botorch.test_functions import Rosenbrock, Ackley, Hartmann, StyblinskiTang
import pickle
from botorch.models.transforms.outcome import Standardize
import matplotlib.pyplot as plt

def synthetic_function(x):
   
    y1 = torch.sin(x[:,0]) * torch.cos(x[:,1]) + x[:,2] * torch.exp(-x[:,0]**2) * torch.cos(x[:,0] + x[:,1]) + 0.01*np.sin((x[:,3] + x[:,4] + x[:,5]))
    y2 = torch.sin(x[:,3]) * torch.cos(x[:,4]) + x[:,5] * torch.exp(-x[:,3]**2) * torch.sin(x[:,3] + x[:,4]) + 0.01*np.cos((x[:,0] + x[:,1] + x[:,2]))
   
    return torch.cat((y1.unsqueeze(1),y2.unsqueeze(1)),dim=1)

def reachability_uniformity(behavior, n_bins = 25, obj_lb1 = -5, obj_ub1 = 5, obj_lb2 = -5, obj_ub2 = 5):
   
    cell_size1 = (obj_ub1 - obj_lb1) / n_bins
    cell_size2 = (obj_ub2 - obj_lb2) / n_bins
    
    indices_y1 = ((behavior[:, 0] - obj_lb1) / cell_size1).floor().int()
    indices_y2 = ((behavior[:, 1] - obj_lb2) / cell_size2).floor().int()
    grid_indices = torch.stack((indices_y1, indices_y2), dim=1)
    unique_cells = set(map(tuple, grid_indices.tolist()))
    return len(unique_cells)/(n_bins**2)
    

if __name__ == '__main__':
    
    lb = -5
    ub = 5
    dim = 6
    N_init = 10
    replicate = 20
    BO_iter = 500
    n_bins = 10
    TS = 1 
    k = 10
    obj_lb1 = -5.1
    obj_ub1 = 5.1
    obj_lb2 = -5.1
    obj_ub2 = 5.1
       
    cost_tensor = []
    coverage_tensor = []
       
    for seed in range(replicate):
        print('seed:',seed)
        np.random.seed(seed)
        
        train_x = torch.tensor(np.random.rand(N_init, dim))
        train_y = synthetic_function((lb+(ub-lb)*train_x))
    
        coverage = reachability_uniformity(train_y, n_bins, obj_lb1, obj_ub1, obj_lb2, obj_ub2)
        
        coverage_list = [coverage]
        cost_list = [0] # number of sampled data excluding initial data
        
        for i in range(BO_iter):        
            
            sobol = SobolEngine(dimension=dim, scramble=True, seed=seed)
            train_x = sobol.draw(n=int(N_init+i+1)).to(torch.float64)
            train_y = synthetic_function((lb+(ub-lb)*train_x))
                       
            coverage = reachability_uniformity(train_y, n_bins, obj_lb1, obj_ub1, obj_lb2, obj_ub2)
            coverage_list.append(coverage)
            cost_list.append(cost_list[-1]+1)
            
        cost_tensor.append(cost_list)
        coverage_tensor.append(coverage_list)
        
    cost_tensor = torch.tensor(cost_tensor, dtype=torch.float32) 
    coverage_tensor = torch.tensor(coverage_tensor, dtype=torch.float32)  
    torch.save(coverage_tensor, 'Cluster_coverage_list_sobol.pt')
    torch.save(cost_tensor, 'Cluster_cost_list_sobol.pt')  
