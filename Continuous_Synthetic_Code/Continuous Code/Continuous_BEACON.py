#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 10 13:26:09 2024

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
import torchsort
import pickle
from botorch.models.transforms.outcome import Standardize
import matplotlib.pyplot as plt
from gpytorch.kernels import RBFKernel, ScaleKernel
from ThompsonSampling import EfficientThompsonSampler
import time


def reachability_uniformity(behavior, n_bins = 25, obj_lb = -5, obj_ub = 5):
    behavior = behavior.squeeze(1).numpy()
    num = len(behavior)
    cum_hist, _ = np.histogram(behavior, np.linspace(obj_lb, obj_ub, n_bins + 1))
    cum_hist = cum_hist[np.nonzero(cum_hist)] / (num) # discrete distribution
    cum_hist_uni = np.mean(cum_hist) * np.ones_like(cum_hist) # theoretical uniform distribution

    cum_coverage = len(cum_hist) / n_bins
    cum_uniformity = 1 - jensenshannon(cum_hist, cum_hist_uni, base=2)
    
    return cum_coverage
    
class CustomAcquisitionFunction(AnalyticAcquisitionFunction):
    def __init__(self, model, sampled_behavior, k=10):
        '''Inits acquisition function with model.'''
        super().__init__(model=model)
        
        self.model = model
        self.k = k
        self.sampled_behavior = sampled_behavior
       
        self.ts_sampler = EfficientThompsonSampler(model)
        self.ts_sampler.create_sample()
        
    @t_batch_mode_transform(expected_q=1)
    def forward(self, X):
        """Compute the acquisition function value at X."""
        
        
        # For other data sets we can sample at once
        # batch_shape x N x m
        # X = X.squeeze(1).unsqueeze(0)
        
        # torch.manual_seed(self.n_seed) # make sure different starting point optimize the same posterior sample
        # posterior = self.model.posterior(X)
        # num_samples x batch_shape x N x m
        # samples = posterior.rsample(sample_shape=torch.Size([1])).squeeze(1) # Thompson Sampling 
        # dist = torch.cdist(samples[0], self.sampled_behavior).squeeze(0)
        
        samples = self.ts_sampler.query_sample(X) # Thompson sampling
        dist = torch.cdist(samples, self.sampled_behavior) # Calculate Euclidean distance between TS and all sampled point
        dist, _ = torch.sort(dist, dim = 1) # sort the distance 
        n = dist.size()[1]
        E = torch.cat((torch.ones(self.k), torch.zeros(n-self.k)), dim = 0) # find the k-nearest neighbor
        dist = dist*E
        acquisition_values = torch.sum(dist,dim=1)
      
        
        return acquisition_values.flatten()


if __name__ == '__main__':
    
    lb = -5 # lower bound for feature
    ub = 5 # upper bound for feature
    dim = 4 # feature dimension
    N_init = 10 # number of initial training data
    replicate = 2# number of replicates for experiment
    BO_iter = 200 # number of evaluations
    n_bins = 25 # grid number for calculating reachability
    TS = 1 # number of TS (posterior sample)
    k = 10 # k-nearest neighbor
    
    # Specify the minimum/maximum value for each synthetic function as to calculate reachability
    
    # obj_lb = 0 # minimum obj value for Rosenbrock
    # obj_ub = 270108 # obj maximum for 4D Rosenbrock
    # obj_ub = 630252.63 # obj maximum for 8D Rosenbrock
    # obj_ub = 990396.990397 # obj maximum for 12D Rosenbrock
    
    obj_lb = 0 # minimum obj value for Ackley
    obj_ub = 14.3027 # maximum obj value for Ackley
    
    # obj_lb = -39.16599*dim # minimum obj val for 4D SkyTang
    # obj_ub = 500 # maximum obj val for 4D SkyTang
    # obj_ub = 1000 # maximum obj val for 8D SkyTang
    # obj_ub = 1500 # maximum obj for 12D SkyTang
   
    # Specify the synthetic function we want to study
    
    # function = Rosenbrock(dim=dim)
    function = Ackley(dim=dim)
    # function = StyblinskiTang(dim=dim)
    
    cost_tensor = []
    coverage_tensor = [] # list containing reachability for every itertation
    time_tensor = [] # list containing CPU time requirement per iteration
    
    for seed in range(replicate): 
        start_time = time.time()
        print('seed:',seed)
        np.random.seed(seed)
        train_x = torch.tensor(np.random.rand(N_init, dim)) # generate initial training data for GP
        
        # sobol = SobolEngine(dimension=dim, scramble=True, seed=seed)
        # train_x = sobol.draw(n=N_init).to(torch.float64)
        
        train_y = function((lb+(ub-lb)*train_x)).unsqueeze(1)
    
        coverage = reachability_uniformity(train_y, n_bins, obj_lb, obj_ub) # Calculate the initial reachability and uniformity    
        coverage_list = [coverage]    
        cost_list = [0] # number of sampled data excluding initial data
             
        # Start BO loop
        for i in range(BO_iter):        
            
            covar_module = ScaleKernel(RBFKernel(ard_num_dims=dim)) # select the RBF kernel
            model = SingleTaskGP(train_x, train_y, outcome_transform=Standardize(m=1), covar_module=covar_module)
            mll = ExactMarginalLogLikelihood(model.likelihood, model)
            fit_gpytorch_mll(mll)
            model.train_x = train_x
            model.train_y = train_y
            
            # Perform optimization on TS posterior sample
            acq_val_max = 0
            for ts in range(TS):
                custom_acq_function = CustomAcquisitionFunction(model, train_y, k=k)
                
                bounds = torch.tensor([[0.0]*dim, [1.0]*dim], dtype=torch.float)  # Define bounds of the feature space (always operate within [0,1]^d)
                # Optimize the acquisition function
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
            
            coverage = reachability_uniformity(train_y, n_bins, obj_lb, obj_ub)
            coverage_list.append(coverage)      
            cost_list.append(cost_list[-1]+1)
  
        end_time = time.time()
        cost_tensor.append(cost_list)
        coverage_tensor.append(coverage_list)
        time_tensor.append((end_time-start_time)/BO_iter)
        
                
    time_tensor = torch.tensor(time_tensor, dtype=torch.float32) 
    cost_tensor = torch.tensor(cost_tensor, dtype=torch.float32) 
    coverage_tensor = torch.tensor(coverage_tensor, dtype=torch.float32)   
    torch.save(coverage_tensor, '4DAckley_coverage_list_NS_TS.pt')
    torch.save(cost_tensor, '4DAckley_cost_list_NS_TS.pt')  
    torch.save(time_tensor, '4DAckley_time_list_NS_TS.pt')
