#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 10 13:26:09 2024

@author: tang.1856
"""
import torch
import gpytorch
from botorch.models import SingleTaskGP, ModelListGP
from gpytorch.kernels import MaternKernel, RBFKernel, ScaleKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.mlls import ExactMarginalLogLikelihood
from torch.quasirandom import SobolEngine
from botorch.fit import fit_gpytorch_mll,fit_fully_bayesian_model_nuts
from botorch.utils import standardize
import botorch
from typing import Tuple
from botorch.utils.transforms import t_batch_mode_transform
from botorch.acquisition import AcquisitionFunction, AnalyticAcquisitionFunction
from botorch.acquisition.multi_objective.analytic import MultiObjectiveAnalyticAcquisitionFunction
from botorch.optim import optimize_acqf
from scipy.spatial.distance import cdist, jensenshannon
import numpy as np
from torch.quasirandom import SobolEngine
import torchsort
import pickle
from botorch.models.transforms.outcome import Standardize
from gpytorch.mlls.sum_marginal_log_likelihood import SumMarginalLogLikelihood
from ThompsonSampling import EfficientThompsonSampler

def synthetic_function(x):
    '''Synthetic multi-output function studied in the paper'''
    y1 = torch.sin(x[:,0]) * torch.cos(x[:,1]) + x[:,2] * torch.exp(-x[:,0]**2) * torch.cos(x[:,0] + x[:,1]) + 0.01*np.sin((x[:,3] + x[:,4] + x[:,5]))
    y2 = torch.sin(x[:,3]) * torch.cos(x[:,4]) + x[:,5] * torch.exp(-x[:,3]**2) * torch.sin(x[:,3] + x[:,4]) + 0.01*np.cos((x[:,0] + x[:,1] + x[:,2]))
    
    return torch.cat((y1.unsqueeze(1),y2.unsqueeze(1)),dim=1)

def reachability_uniformity(behavior, n_bins, obj_lb, obj_ub):
    '''Compute the reachability for given behaviors.'''
    
    indices_list = []
    for n_output in range(N_output):
        cell_size = (obj_ub[n_output] - obj_lb[n_output]) / n_bins
        indices_list.append(((behavior[:, n_output] - obj_lb[n_output]) / cell_size).floor().int())
    
    grid_indices = torch.stack(indices_list,1)
    unique_cells = set(map(tuple, grid_indices.tolist()))
    return len(unique_cells)/(n_bins**2)

class CustomAcquisitionFunction(AcquisitionFunction):
    def __init__(self, model, sampled_behavior, k_NN=10):
        '''Inits acquisition function with model.'''
        super().__init__(model=model)
        
        self.model = model
        self.k_NN = k_NN
        self.sampled_behavior = sampled_behavior
        
        self.ts_sampler_list = []
        for n_output in range(N_output):
            self.ts_sampler_list.append(EfficientThompsonSampler(model.models[n_output]))
            self.ts_sampler_list[n_output].create_sample()
        
    @t_batch_mode_transform(expected_q=1)
    def forward(self, X):
        """Compute the acquisition function value at X."""
        
        sample_list = []
        for n_output in range(N_output):
            sample_list.append(self.ts_sampler_list[n_output].query_sample(X)) # Sample from posterior
        
        samples = torch.cat(sample_list,1)
       
        dist = torch.cdist(samples.to(torch.float64), self.sampled_behavior.to(torch.float64)) # calculate the two norm between each Thompson sampled behavior with sampled behavior
        dist, _ = torch.sort(dist, dim = 1) # sort the distance 
        n = dist.size()[1]
        E = torch.cat((torch.ones(self.k_NN), torch.zeros(n-self.k_NN)), dim = 0) # find the k-nearest neighbor
        dist = dist*E
        acquisition_values = torch.sum(dist,dim=1)
        
        return acquisition_values.flatten()

if __name__ == '__main__':
       
    dim = 6 # dimension for the problem
    N_init = 10 # number of initial data point
    replicate = 20 # number of replicate for experiment
    BO_iter = 500 # number of BO iteration
    n_bins = 10 # number of bins for dividing the outcome space
    N_output = 2 # number of outcome output
    TS = 1 # number of Thompson sampling
    k_NN = 10 # number of k-nearest value
    obj_lb = [-5.1, -5.1] # minimum value for each outcome
    obj_ub = [5.1, 5.1] # maximum value for each outcome
    lb = -5 # lower bound for feature space (assume same lb for all dimension)
    ub = 5 # upper bound for feature space (assume same ub for all dimension)
    
    cost_tensor = []
    coverage_tensor = []
    
    for seed in range(replicate): # each replicate has different initial training data
        print('seed:',seed)
        np.random.seed(seed)
        train_x = torch.tensor(np.random.rand(N_init, dim)) # generate initial training data
        train_y = synthetic_function(lb+(ub-lb)*train_x)

        coverage = reachability_uniformity(train_y, n_bins, obj_lb, obj_ub) # Calculate the initial reachability
        
        coverage_list = [coverage]
        cost_list = [0] # number of sampled data excluding initial data
       
        # Start BO loop
        for i in range(BO_iter):    
            print('Iteration:', i)
            print('Current Reachability = ', max(coverage_list))
            
            # build model list
            model_list = []
            for nx in range(N_output):
                covar_module = ScaleKernel(RBFKernel(ard_num_dims=dim))
                model_list.append(SingleTaskGP(train_x.to(torch.float64), train_y[:,nx].unsqueeze(1).to(torch.float64), outcome_transform=Standardize(m=1), covar_module=covar_module))
            model = ModelListGP(*model_list)
            mll = SumMarginalLogLikelihood(model.likelihood, model)
            
            # Fit the GPs
            try:
                fit_gpytorch_mll(mll)
            except:
                print('Fail to fit GP!')
            
            for nx in range(N_output):
                model.models[nx].train_x = train_x
                model.models[nx].train_y = train_y[:,nx].unsqueeze(1) # Feed the original scale train_y into TS class. It returns the original scale TS back.
           
            custom_acq_function = CustomAcquisitionFunction(model, train_y, k_NN=k_NN) # define our custom acquisition function               
            bounds = torch.tensor([[0.0]*dim, [1.0]*dim], dtype=torch.float) # define the bound for optimizing acquisition function
            
            # Optimize the acquisition function (continuous)
            candidate, acq_value = optimize_acqf(
                acq_function=custom_acq_function,
                bounds=bounds,
                q=1,  # Number of candidates to generate
                num_restarts=10,  # Number of restarts for the optimizer
                raw_samples=20,  # Number of initial raw samples to consider
            )
             
            train_x = torch.cat((train_x, candidate))
            y_new = synthetic_function(lb+(ub-lb)*candidate)                 
            train_y = torch.cat((train_y, y_new))
                       
            coverage = reachability_uniformity(train_y, n_bins, obj_lb, obj_ub) # calculate the current reachability
            coverage_list.append(coverage)
            cost_list.append(cost_list[-1]+1)
                       
        cost_tensor.append(cost_list)
        coverage_tensor.append(coverage_list)
        
    
    cost_tensor = torch.tensor(cost_tensor, dtype=torch.float32) 
    coverage_tensor = torch.tensor(coverage_tensor, dtype=torch.float32) 

    # torch.save(coverage_tensor, 'Cluster_TS_1_coverage_list_NS.pt')
    # torch.save(cost_tensor, 'Cluster_TS_1_cost_list_NS.pt')  
    

