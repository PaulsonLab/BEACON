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
# from botorch import fit_gpytorch_model
import matplotlib.pyplot as plt
import pandas as pd
import math
import itertools
from OilSorbent import OilSorbent
from pyDOE2 import lhs

def reachability_uniformity(xl, xu, behavior):    
    # Number of hypercubes
    N = xl.shape[0]
    
    # Number of dimensions
    D = xl.shape[1]
    
    # Boolean array to keep track of whether a hypercube is filled
    filled = torch.zeros(N, dtype=torch.bool)
    
    # Check if each element in y falls within any of the hypercubes
    for i in range(N):
        lower_bound = xl[i]
        upper_bound = xu[i]
        
        # Check if any element in y is within the current hypercube
        in_hypercube = ((behavior >= lower_bound) & (behavior <= upper_bound)).all(dim=1)
        
        # If any element is within the hypercube, mark it as filled
        if in_hypercube.any():
            filled[i] = True
    
    # Count how many hypercubes are filled
    reachability = filled.sum().item()
    
    # print(f"Number of filled hypercubes: {num_filled_hypercubes}")
    return reachability, filled
    
        
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
    
def partition_bounds(xL, xU, partitions):
    dimensions = len(xL)
    intervals = []

    for i in range(dimensions):
        intervals.append(torch.linspace(xL[i], xU[i], partitions[i] + 1))

    # Generate all combinations of intervals
    lower_bounds = []
    upper_bounds = []
    for index_tuple in itertools.product(*(range(p) for p in partitions)):
        lower_bound = torch.tensor([intervals[dim][index_tuple[dim]] for dim in range(dimensions)])
        upper_bound = torch.tensor([intervals[dim][index_tuple[dim] + 1] for dim in range(dimensions)])
        lower_bounds.append(lower_bound)
        upper_bounds.append(upper_bound)

    # Convert the lists of lower and upper bounds to tensors
    x_LB = torch.stack(lower_bounds)
    x_UB = torch.stack(upper_bounds)
    return x_LB, x_UB
    
if __name__ == '__main__':
    
    dim = 7
    N_init = 10
    replicate = 20
    BO_iter = 200
    n_bins = 10
    TS = 1 
    k_NN = 10
    N_output = 2
    
    # Oil sorbent data set
    fun = OilSorbent(dim)
    lb = torch.tensor([3/7, 0.7, 12, 0.12, 0, 16, 0.41])
    ub = torch.tensor([7/3, 2, 24, 0.18, 0.2, 28, 1.32])
    rand_samp = lhs(dim, 5000, random_state=0) # using latin hypercube
    X_original = torch.tensor(rand_samp, dtype=torch.double)
    Y_original = fun(lb+(ub-lb)*X_original)
    Y_original[Y_original<0] = 0
    
    
    mins = (torch.min(Y_original, dim=0).values)
    maxs = (torch.max(Y_original, dim=0).values)
    
    cut_x = 60000
    cut_y = 6000
    #upper left
    mins1 = torch.tensor([float(mins[0]), cut_y])
    max1 = torch.tensor([cut_x, float(maxs[1])])
    partitions1 = [4,4]
    x_LB1, x_UB1 = partition_bounds(mins1, max1, partitions1)
    
    #lower left
    mins1 = torch.tensor([float(mins[0]), float(mins[1])])
    max1 = torch.tensor([cut_x, cut_y])
    partitions1 = [1,1]
    x_LB2, x_UB2 = partition_bounds(mins1, max1, partitions1)
    
    #lower right
    mins1 = torch.tensor([cut_x, float(mins[1])])
    max1 = torch.tensor([float(maxs[0]), cut_y])
    partitions1 = [4,4]
    x_LB3, x_UB3 = partition_bounds(mins1, max1, partitions1)
    
    #upper right
    mins1 = torch.tensor([cut_x, cut_y])
    max1 = torch.tensor([float(maxs[0]), float(maxs[1])])
    partitions1 = [5,5]
    x_LB4, x_UB4 = partition_bounds(mins1, max1, partitions1)
    
    xl = torch.cat((x_LB1,x_LB2,x_LB3,x_LB4))
    xu = torch.cat((x_UB1,x_UB2,x_UB3,x_UB4))
    
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
        
    coverage_original,_ = reachability_uniformity(xl,xu,Y_original)
    for seed in range(replicate):
        print('seed:',seed)
        np.random.seed(seed)
        ids_acquired = np.random.choice(np.arange((len(Y_original))), size=N_init, replace=False)
        train_x = X_original[ids_acquired]
        train_y = Y_original[ids_acquired]
       
        coverage, filled_total = reachability_uniformity(xl,xu,train_y)
        coverage /=coverage_original      
        coverage_list = [coverage]        
        cost_list = [0]
        
        
        # Start BO loop
        for i in range(BO_iter):        
            
            custom_acq_function = CustomAcquisitionFunction(train_x, k=k_NN)
            
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
            
            coverage, filled_total = reachability_uniformity(xl,xu,train_y)
            coverage /=coverage_original
            coverage_list.append(coverage)            
            cost_list.append(cost_list[-1] + len([id_max_aquisition]))
    
        
        cost_tensor.append(cost_list)
        coverage_tensor.append(coverage_list)
           
    cost_tensor = torch.tensor(cost_tensor, dtype=torch.float32) 
    coverage_tensor = torch.tensor(coverage_tensor, dtype=torch.float32) 
      
    torch.save(coverage_tensor, 'Oil_coverage_list_NS_xspace_120.pt') 
    torch.save(cost_tensor, 'Oil_cost_list_NS_xspace_120.pt')  

    
   
    


