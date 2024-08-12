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
# import torchsort
import pickle
from botorch.models.transforms.outcome import Standardize
import matplotlib.pyplot as plt
import sys
sys.path.append('/home/tang.1856/Jonathan/Novelty Search')
from gpytorch.mlls.sum_marginal_log_likelihood import SumMarginalLogLikelihood
from ThompsonSampling import EfficientThompsonSampler
from sklearn.cluster import KMeans
# import gymnasium as gym
import pandas as pd
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
    return reachability

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
    k_NN = 3
    N_output = 2
    
    # file_path1 = '/home/tang.1856/Downloads/PMOF20K_traindata_7000_train.csv'
    # data1 = pd.read_csv(file_path1)
    # y1 = data1['pure_uptake_CO2_298.00_15000']
    # y2 = data1['pure_uptake_methane_298.00_580000']
    
    # lb = torch.tensor(data1[feat].values.min(axis=0))
    # ub = torch.tensor(data1[feat].values.max(axis=0))
    # X_original = (torch.tensor(data1[feat].values) - lb)/(ub-lb)
    # Y_original = torch.stack((torch.tensor(y1.values),torch.tensor(y2.values)),dim=1)
    np.random.seed(0)
    # ids_acquired_original = np.random.choice(np.arange((len(Y_original))), size=2000, replace=False)
    # X_original = X_original[ids_acquired_original]
    # Y_original = Y_original[ids_acquired_original]
    
    # Oil sorbent data set
    fun = OilSorbent(dim)
    lb = torch.tensor([3/7, 0.7, 12, 0.12, 0, 16, 0.41])
    ub = torch.tensor([7/3, 2, 24, 0.18, 0.2, 28, 1.32])
    rand_samp = lhs(dim, 5000, random_state=0) # using latin hypercube
    X_original = torch.tensor(rand_samp, dtype=torch.double)
    Y_original = fun(lb+(ub-lb)*X_original)
    Y_original[Y_original<0] = 0
    
    
    # obj_lb1 = min(y1.values)
    # obj_ub1 = max(y1.values)
    # obj_lb2 = min(y2.values)
    # obj_ub2 = max(y2.values)
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
    
    # xl, xu = [], []
    # UpperLeft_bin = 2
    # UpperRight_bin = 3
    # LowerRight_bin = 2
    #Upper Left
    # for i in range(UpperLeft_bin):
    #     xl.append([mins[0] + i*(2-mins[0])/UpperLeft_bin, 4 + i*(4-mins[1])/UpperLeft_bin])
    #     xu.append([mins[0] + (i+1)*(2-mins[0])/UpperLeft_bin, 4 + (i+1)*(4-mins[1])/UpperLeft_bin])
    
    # #Upper Right
    # for i in range(UpperRight_bin):
    #     xl.append([2 + i*(maxs[0]-2)/UpperRight_bin, 4 + i*(maxs[1]-4)/UpperRight_bin])
    #     xu.append([2 + (i+1)*(maxs[0]-2)/UpperRight_bin, 4 + (i+1)*(maxs[1]-4)/UpperRight_bin])
    
    # #Lower Right
    # for i in range(LowerRight_bin):
    #     xl.append([2 + i*(maxs[0]-2)/LowerRight_bin, mins[1] + i*(4-mins[1])/LowerRight_bin])
    #     xu.append([2 + (i+1)*(maxs[0]-2)//LowerRight_bin, mins[1] + (i+1)*(4-mins[1])/LowerRight_bin])
    
    
    # xl = torch.tensor(xl)
    # xu = torch.tensor(xu)
    
    
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
    
    # Calculate how many bins are filled by all data points
    coverage_original = reachability_uniformity(xl,xu,Y_original)
    for seed in range(replicate):
        
        print('seed:',seed)
        np.random.seed(seed)
        ids_acquired = np.random.choice(np.arange((len(Y_original))), size=N_init, replace=False)
       
        train_x = X_original[ids_acquired]
        train_y = Y_original[ids_acquired]
        
        coverage = reachability_uniformity(xl,xu,train_y)/coverage_original
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
                       
            coverage = reachability_uniformity(xl,xu,train_y)/coverage_original
            coverage_list.append(coverage)           
            cost_list.append(cost_list[-1]+1)
           
        cost_tensor.append(cost_list)
        coverage_tensor.append(coverage_list)
      
    
    cost_tensor = torch.tensor(cost_tensor, dtype=torch.float32) 
    coverage_tensor = torch.tensor(coverage_tensor, dtype=torch.float32) 
   
    torch.save(cost_tensor, 'Oil_cost_list_BEACON_120.pt')  
    torch.save(coverage_tensor, 'Oil_coverage_list_BEACON_120.pt')  
