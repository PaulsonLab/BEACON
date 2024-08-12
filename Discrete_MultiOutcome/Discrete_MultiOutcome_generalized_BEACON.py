#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 10 13:26:09 2024

@author: tang.1856
"""
import torch
import gpytorch
from botorch.models import SingleTaskGP, SaasFullyBayesianSingleTaskGP, ModelListGP
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
from botorch.test_functions import Rosenbrock, Ackley, Hartmann, StyblinskiTang
import torchsort
import pickle
from botorch.models.transforms.outcome import Standardize
import matplotlib.pyplot as plt
import sys
sys.path.append('/home/tang.1856/Jonathan/Novelty Search')
from gpytorch.mlls.sum_marginal_log_likelihood import SumMarginalLogLikelihood
from ThompsonSampling import EfficientThompsonSampler
from sklearn.cluster import KMeans
import gymnasium as gym
import pandas as pd

def reachability_uniformity(behavior, n_bins = 25, mins = None, maxs=None, num_filled_grids = 100):
    
    
    filled_grids = set()
    grid_sizes = (maxs - mins) / (n_bins)
    # Assign points to grids
    for point in behavior:
        if all(point>=maxs):
            grid_indices = torch.tensor([n_bins-1]*N_output)
        else:
            grid_indices = ((point - mins) / grid_sizes).int()
        # Use a tuple (immutable) to store grid indices in the set
        filled_grids.add(tuple(grid_indices.tolist()))
        
        
    return len(filled_grids)/(num_filled_grids)

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
            sample_list.append(self.ts_sampler_list[n_output].query_sample(X))
        
        samples = torch.cat(sample_list,1)
        
        dist = torch.cdist(samples.to(torch.float64), self.sampled_behavior.to(torch.float64))
        dist, _ = torch.sort(dist, dim = 1) # sort the distance 
        n = dist.size()[1]
        E = torch.cat((torch.ones(self.k_NN), torch.zeros(n-self.k_NN)), dim = 0) # find the k-nearest neighbor
        dist = dist*E
        acquisition_values = torch.sum(dist,dim=1)
        
        return acquisition_values.flatten()



if __name__ == '__main__':
    
    # Case study 1
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
    
    # Case study 2
    # df = pd.read_csv('/home/tang.1856/Jonathan/Novelty Search/Training Data/rawdata/Nitrogen.csv') # data from Boobier et al.
    # X_original = (df.iloc[:, 1:21]).values
    # X_original = torch.from_numpy(X_original)    
    # X_original = (X_original - X_original.min(dim=0).values)/(X_original.max(dim=0).values - X_original.min(dim=0).values) 
    # y1 = df['U_N2 (mol/kg)']
    # y2 = df['D_N2 (cm2/s)']   
    # Y_original = torch.stack((torch.tensor(y1.values),torch.tensor(y2.values)),dim=1)
    
    dim = 25
    N_init = 50
    replicate = 20
    BO_iter = 500
    n_bins = 10
    TS = 1 
    k_NN = 10
    N_output = 2
    
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
            grid_indices = torch.tensor([n_bins-1]*N_output)
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
        ids_acquired = np.random.choice(np.arange((len(y1))), size=N_init, replace=False)
       
        train_x = X_original[ids_acquired]
        train_y = Y_original[ids_acquired]
        
        coverage = reachability_uniformity(train_y, n_bins, mins, maxs, num_filled_grids)   
        coverage_list = [coverage]        
        cost_list = [0] 
      
        
        # Start BO loop
        for i in range(BO_iter):        
            print('Iteration:', i)
            print('Current Reachability = ', max(coverage_list))
            
            model_list = []
            for nx in range(N_output):
                covar_module = ScaleKernel(RBFKernel(ard_num_dims=dim))
                model_list.append(SingleTaskGP(train_x.to(torch.float64), train_y[:,nx].unsqueeze(1).to(torch.float64), outcome_transform=Standardize(m=1), covar_module=covar_module))
            model = ModelListGP(*model_list)
            mll = SumMarginalLogLikelihood(model.likelihood, model)
            
            try:
                fit_gpytorch_mll(mll)
            except:
                print('Fail to fit GP!')
            
            for nx in range(N_output):
                model.models[nx].train_x = train_x
                model.models[nx].train_y = train_y[:,nx].unsqueeze(1)
            
            # Perform optimization on TS posterior sample            
            custom_acq_function = CustomAcquisitionFunction(model, train_y, k_NN=k_NN)
                
            # Optimize the acquisition function (discrete)
            acquisition_values = custom_acq_function(X_original.unsqueeze(1).to(torch.float32)) # calculate acquisition value for all data point
            ids_sorted_by_aquisition = acquisition_values.argsort(descending=True)
            for id_max_aquisition_all in ids_sorted_by_aquisition: 
                if not id_max_aquisition_all.item() in ids_acquired: # make sure we don't re-sample point
                    id_max_aquisition = id_max_aquisition_all.item()
                    break
            
            ids_acquired = np.concatenate((ids_acquired, [id_max_aquisition]))
            train_x = X_original[ids_acquired]
            train_y = Y_original[ids_acquired]
                       
            coverage = reachability_uniformity(train_y, n_bins, mins, maxs, num_filled_grids)
            coverage_list.append(coverage)           
            cost_list.append(cost_list[-1]+1)
           
        cost_tensor.append(cost_list)
        coverage_tensor.append(coverage_list)
      
    
    cost_tensor = torch.tensor(cost_tensor, dtype=torch.float32) 
    coverage_tensor = torch.tensor(coverage_tensor, dtype=torch.float32) 
   
    # torch.save(cost_tensor, 'MOF_TS_1_cost_list_NS.pt')  
    # torch.save(coverage_tensor, 'MOF_TS_1_coverage_list_NS.pt')  
