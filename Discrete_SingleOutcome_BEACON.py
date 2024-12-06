#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 10 13:26:09 2024

@author: tang.1856
"""
import torch
from botorch.models import SingleTaskGP
from gpytorch.kernels import MaternKernel, RFFKernel, ScaleKernel
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.fit import fit_gpytorch_mll
from scipy.spatial.distance import cdist, jensenshannon
import numpy as np
from botorch.models.transforms.outcome import Standardize
import pandas as pd

def reachability(behavior, n_bins = 25, obj_lb = -5, obj_ub = 5, n_hist = 25):
    behavior = behavior.squeeze(1).numpy()
    num = len(behavior)
    cum_hist, _ = np.histogram(behavior, np.linspace(obj_lb, obj_ub, n_bins + 1))
    cum_hist = cum_hist[np.nonzero(cum_hist)] / (num) # discrete distribution
    cum_coverage = len(cum_hist) / n_hist  
    return cum_coverage
    
class CustomAcquisitionFunction_TS():
    def __init__(self, model, sampled_behavior, k=10, TS = 1):
        
        self.model = model
        self.k = k
        self.sampled_behavior = sampled_behavior
        self.TS = TS
    
    def __call__(self, X):
      
        # batch_shape x N x m
        X = X.unsqueeze(0)
        posterior = self.model.posterior(X)
        
        # num_samples x batch_shape x N x m
        samples = posterior.rsample(sample_shape=torch.Size([self.TS])).squeeze(1) # Thompson Sampling
        
        acquisition_values = []
        for ts in range(self.TS): 
            
            dist = torch.cdist(samples[ts], self.sampled_behavior).squeeze(0)
            dist, _ = torch.sort(dist, dim = 1) # sort the distance 
            n = dist.size()[1]
            E = torch.cat((torch.ones(self.k), torch.zeros(n-self.k)), dim = 0) # find the k-nearest neighbor
            dist = dist*E
            acquisition_values.append(torch.sum(dist,dim=1))
            
        acquisition_values = torch.stack(acquisition_values)
        acquisition_values = torch.max(acquisition_values, dim=0).values
               
        return acquisition_values.flatten()
    

if __name__ == '__main__':
    
    dim = 14 # dimension for the input feature (molecular descriptor)
    N_init = 10 # number of initial training data for GP
    replicate = 20
    n_bins = 25 
    k_NN = 10 
    TS = 1 
    BO_iter = 100 # number of function evaluation
    
    cost_tensor = []
    coverage_tensor = []
   
    # Case Study
    # load water solubility data from csv file 
    df = pd.read_csv('/home/tang.1856/BEACON/BEACON/Material Data/water_set_wide_descriptors.csv') # data from Boobier et al.
    X_original = (df.iloc[:, 4:(4+dim)]).values
    y_original = df.iloc[:, 3].values
        
    y_original = np.reshape(y_original, (np.size(y_original), 1)) 
    X_original = torch.from_numpy(X_original)
    
    X_original = (X_original - X_original.min(dim=0).values)/(X_original.max(dim=0).values - X_original.min(dim=0).values) # normalize the original input data
    y_original = torch.from_numpy(y_original)
    
    n_hist = float(torch.count_nonzero(torch.histc(y_original,bins=n_bins)))
    obj_lb = y_original.min() # obj minimum
    obj_ub = y_original.max() # obj maximum 
    
    for seed in range(replicate):
        print('seed:',seed)
        np.random.seed(seed)
        
        # randomly select initial training data
        ids_acquired = np.random.choice(np.arange((len(y_original))), size=N_init, replace=False)
        train_x = X_original[ids_acquired]
        train_y = y_original[ids_acquired]
        
        # Calculate the initial reachability
        coverage = reachability(train_y, n_bins, obj_lb, obj_ub, n_hist)
        coverage_list = [coverage]
    
        # Start BEACON loop
        for i in range(BO_iter):    
            
            print('iteration:', i)
            
            model = SingleTaskGP(train_x, train_y, outcome_transform=Standardize(m=1))
            mll = ExactMarginalLogLikelihood(model.likelihood, model)
            fit_gpytorch_mll(mll)
            
            custom_acq_function = CustomAcquisitionFunction_TS(model, train_y, k=k_NN, TS = TS)
            
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
            
            coverage = reachability(train_y, n_bins, obj_lb, obj_ub, n_hist) # Calculate the new reachability
            coverage_list.append(coverage)
            
        coverage_tensor.append(coverage_list)
           
    coverage_tensor = torch.tensor(coverage_tensor, dtype=torch.float32) 
   
    
    
    


