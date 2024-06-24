#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  6 19:19:01 2024

@author: tang.1856
"""


import torch
import gpytorch
from botorch.models import SingleTaskGP
from gpytorch.kernels import MaternKernel, RBFKernel, ScaleKernel
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

class toy_function():
    def __init__(self, dim):
        self.dim = dim
        
    def __call__(self, x):
        # fun_val = -5 * torch.sin((2*torch.pi * x[:,0]) / (11-1.8*x[:,1]))
        fun_val = 5 * torch.cos(torch.pi / 2 + 2 * torch.pi * x[:, 0] / (11 - 1.8 * x[:, 1])) # Toy problem studied in the paper
    
        return fun_val

def reachability_uniformity(behavior, n_bins = 25, obj_lb = -5, obj_ub = 5):
    behavior = behavior.squeeze(1).numpy()
    num = len(behavior)
    cum_hist, _ = np.histogram(behavior, np.linspace(obj_lb, obj_ub, n_bins + 1))
    cum_hist = cum_hist[np.nonzero(cum_hist)] / (num) # discrete distribution
    cum_hist_uni = np.mean(cum_hist) * np.ones_like(cum_hist) # theoretical uniform distribution

    cum_coverage = len(cum_hist) / n_bins
    cum_uniformity = 1 - jensenshannon(cum_hist, cum_hist_uni, base=2)
    
    return cum_coverage, cum_uniformity
    
class CustomAcquisitionFunction_Mean(AnalyticAcquisitionFunction):
    def __init__(self, model, sampled_behavior, k=10):
        '''Inits acquisition function with model.'''
        super().__init__(model=model)
        
        self.model = model
        self.k = k
        self.sampled_behavior = sampled_behavior
     
        
    @t_batch_mode_transform(expected_q=1)
    def forward(self, X):
        """Compute the acquisition function value at X."""
        
        # Obtain model predictions
        posterior = self.model.posterior(X)
        # samples = posterior.rsample(sample_shape=torch.Size([1])) # Thompson Sampling
        samples = posterior.mean.unsqueeze(0)
        dist = torch.norm(samples - self.sampled_behavior, dim=0)
        dist, _ = torch.sort(dist, dim = 1) # sort the distance 
        n = dist.size()[1]

        E = torch.cat((torch.ones(self.k), torch.zeros(n-self.k)), dim = 0) # find the k-nearest neighbor
        dist = dist*E.unsqueeze(0).unsqueeze(-1)
        acquisition_values = torch.sum(dist,dim=1)
        
        return acquisition_values.flatten()


if __name__ == '__main__':
    lb = -5
    ub = 5
    dim = 8
    N_init = 10
    replicate = 20
    BO_iter = 200
    n_bins = 25
    k = 10
    
    obj_lb = 0
    # obj_lb = -5 # minimum obj value for toy function
    # obj_ub = 270108 # obj maximum for Rosenbrock
    # obj_ub = 630252.63 # obj maximum for 8D Rosenbrock
    # obj_ub = 990396.990397 # obj maximum for 12D Rosenbrock
    obj_ub = 14.3027 # maximum obj value for Ackley
    # obj_ub = 5
    # obj_lb = -3.3224 # minimum for 6D Hartmann
    # obj_ub = 0 # maximum for 6D Hartmann
    # obj_lb = -39.16599*dim # minimum obj val for SkyTang
    # obj_ub = 500 # maximum obj val for 4D SkyTang
    # obj_ub = 1000 # maximum obj val for 8D SkyTang
    # obj_ub = 1500 # maximum obj val for 12D StyTang
    cost_tensor = []
    coverage_tensor = []
    uniformity_tensor = []
    
    
    for seed in range(replicate):
        print('seed:',seed)
        np.random.seed(seed)
        
        # train_x = torch.tensor((np.random.rand(N_init, dim)-0.5-lb)/(ub-lb)) # generate the same intial data as the evolutionay-based NS
        train_x = torch.tensor(np.random.rand(N_init, dim))
        # train_x = torch.rand(20,2).to(torch.float64)
        # sobol = SobolEngine(dimension=dim, scramble=True, seed=seed)
        # train_x = 0.4+0.2*sobol.draw(n=N_init).to(torch.float64)
        
        # function = toy_function(dim=dim)
        # function = Rosenbrock(dim=dim)
        function = Ackley(dim=dim)
        # function = Hartmann(dim=dim)
        # function = StyblinskiTang(dim=dim)
        
        # best_reward = max(train_y)
        train_y = function((lb+(ub-lb)*train_x)).unsqueeze(1)
    
        coverage, uniformity = reachability_uniformity(train_y, n_bins, obj_lb, obj_ub) # Calculate the initial reachability and uniformity
        
        coverage_list = [coverage]
        uniformity_list = [uniformity]
        cost_list = [0] # number of sampled data excluding initial data
        
        
        # Start BO loop
        for i in range(BO_iter):        
            
            # covar_module = ScaleKernel(base_kernel)
            covar_module = ScaleKernel(RBFKernel(ard_num_dims=dim))
            model = SingleTaskGP(train_x, train_y, outcome_transform=Standardize(m=1), covar_module=covar_module)
            mll = ExactMarginalLogLikelihood(model.likelihood, model)
            fit_gpytorch_mll(mll)
            
            custom_acq_function = CustomAcquisitionFunction_Mean(model, train_y, k=k)
            
            bounds = torch.tensor([[0.0]*dim, [1.0]*dim], dtype=torch.float)  # Define bounds of the parameter space
            
            # Optimize the acquisition function (continuous)
            candidate, acq_value = optimize_acqf(
                acq_function=custom_acq_function,
                bounds=bounds,
                q=1,  # Number of candidates to generate
                num_restarts=5,  # Number of restarts for the optimizer
                raw_samples=20,  # Number of initial raw samples to consider
            )
            
            train_x = torch.cat((train_x, candidate))
            y_new = function(lb+(ub-lb)*candidate).unsqueeze(1)
            train_y = torch.cat((train_y, y_new))
            
            coverage, uniformity = reachability_uniformity(train_y, n_bins, obj_lb, obj_ub)
            coverage_list.append(coverage)
            uniformity_list.append(uniformity)
            cost_list.append(cost_list[-1]+1)
            
            # if y_new>best_reward:
            #     print('Best Found reward = ', max(train_y))
            #     best_reward = y_new
        
        cost_tensor.append(cost_list)
        coverage_tensor.append(coverage_list)
        uniformity_tensor.append(uniformity_list)
    
        # calculate_rmse(train_x, train_y)
    
    cost_tensor = torch.tensor(cost_tensor, dtype=torch.float32) 
    coverage_tensor = torch.tensor(coverage_tensor, dtype=torch.float32) 
    uniformity_tensor = torch.tensor(uniformity_tensor, dtype=torch.float32)  
    torch.save(coverage_tensor, '8DAckley_coverage_list_NS_mean.pt')
    # torch.save(uniformity_tensor, '4DAckley_uniformity_list_NS_mean.pt')
    torch.save(cost_tensor, '8DAckley_cost_list_NS_mean.pt')  
