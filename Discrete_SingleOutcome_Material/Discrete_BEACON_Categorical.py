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
from botorch.models.kernels.categorical import CategoricalKernel

device = torch.device('cpu')
print(device)

def reachability_uniformity(behavior, n_bins = 25, obj_lb = -5, obj_ub = 5, n_hist = 25):
    behavior = behavior.cpu().squeeze(1).numpy()
    num = len(behavior)
    cum_hist, _ = np.histogram(behavior, np.linspace(obj_lb.cpu(), obj_ub.cpu(), n_bins + 1))
    cum_hist = cum_hist[np.nonzero(cum_hist)] / (num) # discrete distribution
    cum_hist_uni = np.mean(cum_hist) * np.ones_like(cum_hist) # theoretical uniform distribution

    cum_coverage = len(cum_hist) / n_hist
    cum_uniformity = 1 - jensenshannon(cum_hist, cum_hist_uni, base=2)
    
    return cum_coverage, cum_uniformity
    

class CustomAcquisitionFunction_TS():
    def __init__(self, model, sampled_behavior, k=10, TS = 1):
        
        self.model = model
        self.k = k
        self.sampled_behavior = sampled_behavior
        self.TS = TS
    
    def __call__(self, X):
        """Compute the acquisition function value at X.
        Args:
            X: A `N x d`-dim Tensor from which to sample (in the `N`
                dimension)
            num_samples: The number of samples to draw.
            
        """
              
        # For H2 data set we might need to sample with for loop because of the large data set that might crash the memory (uncommented the following lines)
        # batch_shape x N x m
        # X = X.unsqueeze(0)
        # posterior = self.model.posterior(X[:,0:1000])
        # samples = posterior.rsample(sample_shape=torch.Size([self.TS])) # Thompson Sampling
        # for r in range(1,math.ceil(len(X[0])/1000)):           
        #     posterior = self.model.posterior(X[:, int(r*1000):int((r+1))*1000])
        #     samples_new = posterior.rsample(sample_shape=torch.Size([self.TS])) # Thompson Sampling
        #     samples = torch.cat((samples, samples_new), dim=2)
        # samples = samples.squeeze(1)
        
        # For other data sets we can sample at once
        # batch_shape x N x m
        X = X.unsqueeze(0)
        posterior = self.model.posterior(X)
        
        # num_samples x batch_shape x N x m
        samples = posterior.rsample(sample_shape=torch.Size([self.TS])).squeeze(1) # Thompson Sampling
        
        acquisition_values = []
        for ts in range(self.TS): # different TS sample
            
            dist = torch.cdist(samples[ts], self.sampled_behavior).squeeze(0)
            dist, _ = torch.sort(dist, dim = 1) # sort the distance 
            n = dist.size()[1]
            E = torch.cat((torch.ones(self.k), torch.zeros(n-self.k)), dim = 0).to(device) # find the k-nearest neighbor
            dist = dist*E
            acquisition_values.append(torch.sum(dist,dim=1))
            
        acquisition_values = torch.stack(acquisition_values)
        acquisition_values = torch.max(acquisition_values, dim=0).values
        
        
        return acquisition_values.flatten()
    

if __name__ == '__main__':
    
    dim = 8
    N_init = 50
    replicate = 5
    n_bins = 50
    k = 10 
    TS = 1 
    BO_iter = 300
    
    cost_tensor = []
    coverage_tensor = []
    
    np.random.seed(0)
    
    indices = np.arange(65729)
    # ids_acquired_original = np.random.choice(indices, 20000, replace=False)
    ids_acquired_original = np.random.choice(np.arange((65792)), size=10000, replace=False)
    X_original= np.load('/home/tang.1856/BEACON/BEACON/Discrete_Material_Code/data/tf_bind_8-SIX6_REF_R1/tf_bind_8-x-0.npy')[ids_acquired_original]
    y_original = np.load('/home/tang.1856/BEACON/BEACON/Discrete_Material_Code/data/tf_bind_8-SIX6_REF_R1/tf_bind_8-y-0.npy')[ids_acquired_original]
      
    
    y_original = np.reshape(y_original, (np.size(y_original), 1)) 
    X_original = torch.from_numpy(X_original).to(torch.float64).to(device)
    
    X_original = (X_original - X_original.min(dim=0).values)/(X_original.max(dim=0).values - X_original.min(dim=0).values) # normalize the original input data
    y_original = torch.from_numpy(y_original).to(torch.float64).to(device)
    
    for seed in range(1,replicate):
        print('seed:',seed)
        
        
        n_hist = float(torch.count_nonzero(torch.histc(y_original,bins=n_bins)))
        obj_lb = y_original.min() # obj minimum
        obj_ub = y_original.max() # obj maximum 
        
        np.random.seed(seed)
        ids_acquired = np.random.choice(np.arange((len(y_original))), size=N_init, replace=False)
        train_x = X_original[ids_acquired]
        train_y = y_original[ids_acquired]
       
        coverage, uniformity = reachability_uniformity(train_y, n_bins, obj_lb, obj_ub, n_hist)
        
        coverage_list = [coverage]
        cost_list = [0]
    
        # Start BO loop
        for i in range(BO_iter):  
            torch.cuda.empty_cache()
            covar_module = CategoricalKernel(ard_num_dims = dim)
            model = SingleTaskGP(train_x, train_y, covar_module = covar_module, outcome_transform=Standardize(m=1))
            mll = ExactMarginalLogLikelihood(model.likelihood, model)
            
            try:
                fit_gpytorch_mll(mll)
            except:
                print('cant fit GP!')
            
            custom_acq_function = CustomAcquisitionFunction_TS(model, train_y, k=k, TS = TS)
            
            # Optimize the acquisition function (discrete)
            acquisition_values = custom_acq_function(X_original)
            ids_sorted_by_aquisition = acquisition_values.argsort(descending=True)
            for id_max_aquisition_all in ids_sorted_by_aquisition:
                if not id_max_aquisition_all.item() in ids_acquired:
                    id_max_aquisition = id_max_aquisition_all.item()
                    break
                
            ids_acquired = np.concatenate((ids_acquired, [id_max_aquisition]))
            train_x = X_original[ids_acquired]
            train_y = y_original[ids_acquired] 
            
            coverage, uniformity = reachability_uniformity(train_y, n_bins, obj_lb, obj_ub, n_hist)
            coverage_list.append(coverage)
            cost_list.append(cost_list[-1] + len([id_max_aquisition]))
            
        cost_tensor.append(cost_list)
        coverage_tensor.append(coverage_list)
           
    cost_tensor = torch.tensor(cost_tensor, dtype=torch.float32) 
    coverage_tensor = torch.tensor(coverage_tensor, dtype=torch.float32) 
    torch.save(coverage_tensor, 'DNA_TS_1_coverage_list_NS1.pt')
    torch.save(cost_tensor, 'DNA_TS_1_cost_list_NS1.pt')  
    
    
    


