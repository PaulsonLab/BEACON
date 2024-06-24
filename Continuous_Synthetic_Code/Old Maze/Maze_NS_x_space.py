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
import sys
sys.path.append('/home/tang.1856/Jonathan/Hands-on-Neuroevolution-with-Python-master')
from maze_NS import Maze
from torch.quasirandom import SobolEngine

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
    
class CustomAcquisitionFunction(AnalyticAcquisitionFunction):
    def __init__(self, model, sampled_X, k=10, n_seed = 0):
        '''Inits acquisition function with model.'''
        super().__init__(model=model)
        
        self.model = model
        self.k = k
        self.sampled_X = sampled_X
        self.n_seed = n_seed
        
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
    
    X_opt = -30+60*torch.tensor([[0.0029, 0.2438, 0.3485, 0.5100, 0.6259, 0.1782, 0.1210, 0.1420, 0.9787,
            0.6091, 0.7387, 0.4671, 0.8452, 0.5360, 0.4084, 0.0163, 0.3617, 0.4038,
            0.3541, 0.2491, 0.8930, 0.6666]])
    
    lb = X_opt-7
    ub = X_opt+7
    
    dim = 22
    N_init = 30
    replicate = 20
    BO_iter = 300
    n_bins = 25
    TS = 1 # number of TS (posterior sample)
    k = 10 # k-nearest neighbor
    obj_lb = 0
   
    obj_ub = 1 # maximum for 6D Hartmann
    
    cost_tensor = []
    coverage_tensor = []
    uniformity_tensor = []
    cumbent_tensor = []
    
    for seed in range(replicate):
        print('seed:',seed)
        np.random.seed(seed)
        
        # train_x = torch.tensor((np.random.rand(N_init, dim)-0.5-lb)/(ub-lb)) # generate the same intial data as the evolutionay-based NS
        # train_x = torch.tensor(np.random.rand(N_init, dim))
        # train_x = torch.rand(20,2).to(torch.float64)
        sobol = SobolEngine(dimension=dim, scramble=True, seed=seed)
        train_x = sobol.draw(n=N_init).to(torch.float64)
        
        train_y = []
        for k in range(N_init):
            fun = Maze()
            train_y.append(fun.run_experiment(lb+(ub-lb)*train_x[k].unsqueeze(0)))
         
        train_y = torch.tensor(train_y).unsqueeze(1)
        
        # best_reward = max(train_y)
        
    
        coverage, uniformity = reachability_uniformity(train_y, n_bins, obj_lb, obj_ub) # Calculate the initial reachability and uniformity
        
        coverage_list = [coverage]
        uniformity_list = [uniformity]
        cost_list = [0] # number of sampled data excluding initial data
        cumbent_list = [float(max(train_y))]
        
        # Start BO loop
        for i in range(BO_iter):        
            
           
            model = SingleTaskGP(train_x.to(torch.float64), train_y.to(torch.float64), outcome_transform=Standardize(m=1))
            mll = ExactMarginalLogLikelihood(model.likelihood, model)
            # fit_gpytorch_mll(mll)
            
            
            # Perform optimization on TS posterior sample
            acq_val_max = 0
            for ts in range(TS):
                custom_acq_function = CustomAcquisitionFunction(model, train_x, k=k, n_seed=torch.randint(low=0,high=int(1e10),size=(1,)))
                
                bounds = torch.tensor([[0.0]*dim, [1.0]*dim], dtype=torch.float)  # Define bounds of the parameter space
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
            fun = Maze()
            y_new = fun.run_experiment(lb+(ub-lb)*candidate)
            train_y = torch.cat((train_y, y_new))
            
            coverage, uniformity = reachability_uniformity(train_y, n_bins, obj_lb, obj_ub)
            coverage_list.append(coverage)
            uniformity_list.append(uniformity)
            cost_list.append(cost_list[-1]+1)
            cumbent_list.append(float(max(train_y)))
            torch.save(train_x, 'train_x_NS_xspace_seed'+str(seed)+'.pt')
            torch.save(train_y, 'train_y_NS_xspace_seed'+str(seed)+'.pt')
            # if y_new>best_reward:
            #     print('Best Found reward = ', max(train_y))
            #     best_reward = y_new
        
        cost_tensor.append(cost_list)
        coverage_tensor.append(coverage_list)
        uniformity_tensor.append(uniformity_list)
        cumbent_tensor.append(cumbent_list)
        
    
    cumbent_tensor = torch.tensor(cumbent_tensor, dtype=torch.float32) 
    cost_tensor = torch.tensor(cost_tensor, dtype=torch.float32) 
    coverage_tensor = torch.tensor(coverage_tensor, dtype=torch.float32) 
    uniformity_tensor = torch.tensor(uniformity_tensor, dtype=torch.float32)  
    torch.save(coverage_tensor, 'MediumMaze_coverage_list_NS_x_space.pt')
    # torch.save(uniformity_tensor, 'MediumMaze_uniformity_list_NS_x_space.pt')
    torch.save(cost_tensor, 'MediumMaze_cost_list_NS_x_space.pt')  
    torch.save(cumbent_tensor, 'MediumMaze_cumbent_list_NS_x_space.pt')  
