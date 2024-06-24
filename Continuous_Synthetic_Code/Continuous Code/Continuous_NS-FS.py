#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 11 13:31:13 2024

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
import time

def reachability_uniformity(behavior, n_bins = 25, obj_lb = -5, obj_ub = 5):
    behavior = behavior.squeeze(1).numpy()
    num = len(behavior)
    cum_hist, _ = np.histogram(behavior, np.linspace(obj_lb, obj_ub, n_bins + 1))
    cum_hist = cum_hist[np.nonzero(cum_hist)] / (num) # discrete distribution
    cum_hist_uni = np.mean(cum_hist) * np.ones_like(cum_hist) # theoretical uniform distribution

    cum_coverage = len(cum_hist) / n_bins
    cum_uniformity = 1 - jensenshannon(cum_hist, cum_hist_uni, base=2)
    
    return cum_coverage, cum_uniformity
    
class CustomAcquisitionFunction(AnalyticAcquisitionFunction):
    def __init__(self, model, sampled_X, k=10):
        '''Inits acquisition function with model.'''
        super().__init__(model=model)
        
        self.model = model
        self.k = k
        self.sampled_X = sampled_X
        
    @t_batch_mode_transform(expected_q=1)
    def forward(self, X):
        """Compute the acquisition function value at X."""
        
        dist = torch.norm(X - self.sampled_X, dim=2)
        dist, _ = torch.sort(dist, dim = 1) # sort the distance 
        n = dist.size()[1]

        E = torch.cat((torch.ones(self.k), torch.zeros(n-self.k)), dim = 0) # find the k-nearest neighbor
        dist = dist*E
        acquisition_values = torch.sum(dist,dim=1)
        
        return acquisition_values.flatten()

if __name__ == '__main__':
    
    lb = -5
    ub = 5
    dim = 4
    N_init = 10
    replicate = 20
    BO_iter = 200
    n_bins = 25
    TS = 1 # number of TS (posterior sample)
    k = 10 # k-nearest neighbor
    
    # Specify the minimum/maximum value for each synthetic function as to calculate reachability
    
    obj_lb = 0 # minimum obj value for Rosenbrock
    obj_ub = 270108 # obj maximum for 4D Rosenbrock
    # obj_ub = 630252.63 # obj maximum for 8D Rosenbrock
    # obj_ub = 990396.990397 # obj maximum for 12D Rosenbrock
    # obj_ub = 1710685.71
    
    # obj_lb = 0 # minimum obj value for Ackley
    # obj_ub = 14.3027 # maximum obj value for Ackley
    
    # obj_lb = -39.16599*dim # minimum obj val for 4D SkyTang
    # obj_ub = 500 # maximum obj val for 4D SkyTang
    # obj_ub = 1000 # maximum obj val for 8D SkyTang
    # obj_ub = 1500 # maximum obj for 12D SkyTang
   
    # Specify the synthetic function we want to study
    
    function = Rosenbrock(dim=dim)
    # function = Ackley(dim=dim)
    # function = StyblinskiTang(dim=dim)
    
    cost_tensor = []
    coverage_tensor = []
    uniformity_tensor = []
    
    
    for seed in range(replicate):
        print('seed:',seed)
        np.random.seed(seed)       
        train_x = torch.tensor(np.random.rand(N_init, dim))
        
        train_y = function((lb+(ub-lb)*train_x)).unsqueeze(1)    
        coverage, uniformity = reachability_uniformity(train_y, n_bins, obj_lb, obj_ub) # Calculate the initial reachability and uniformity
        
        coverage_list = [coverage]
        cost_list = [0] # number of sampled data excluding initial data
        
        time_tensor = []
        # Start BO loop
        for i in range(BO_iter):        
            start_time = time.time()
            # We actually don't need the GP
            model = SingleTaskGP(train_x, train_y, outcome_transform=Standardize(m=1))
            mll = ExactMarginalLogLikelihood(model.likelihood, model)
            # fit_gpytorch_mll(mll)
            
            
            # Perform optimization
            acq_val_max = 0
            for ts in range(TS):
                custom_acq_function = CustomAcquisitionFunction(model, train_x, k=k)
                
                bounds = torch.tensor([[0.0]*dim, [1.0]*dim], dtype=torch.float)  # Define bounds of the feature space
                # Optimize the acquisition function (continuous)
                candidate_ts, acq_value = optimize_acqf(
                    acq_function=custom_acq_function,
                    bounds=bounds,
                    q=1,  # Number of candidates to generate
                    num_restarts=10,  # Number of restarts for the optimizer
                    raw_samples=20,  # Number of initial raw samples to consider
                )
                
                if acq_value>acq_val_max:
                    acq_val_max = acq_value
                    candidate = candidate_ts
            
            train_x = torch.cat((train_x, candidate))
            y_new = function(lb+(ub-lb)*candidate).unsqueeze(1)
            train_y = torch.cat((train_y, y_new))
            
            coverage, uniformity = reachability_uniformity(train_y, n_bins, obj_lb, obj_ub)
            coverage_list.append(coverage)
            cost_list.append(cost_list[-1]+1)           
         
        end_time = time.time()
        time_tensor.append((end_time-start_time)/BO_iter)
        cost_tensor.append(cost_list)
        coverage_tensor.append(coverage_list)
    
    cost_tensor = torch.tensor(cost_tensor, dtype=torch.float32) 
    coverage_tensor = torch.tensor(coverage_tensor, dtype=torch.float32)   
    time_tensor = torch.tensor(time_tensor, dtype=torch.float32) 
    torch.save(time_tensor, '4DRosen_time_list_NS_x_space.pt')
    torch.save(coverage_tensor, '4DRosen_coverage_list_NS_x_space.pt')
    torch.save(cost_tensor, '4DRosen_cost_list_NS_x_space.pt')  
