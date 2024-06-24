#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 11 13:25:40 2024

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

def reachability_uniformity(behavior, n_bins = 25, obj_lb = -5, obj_ub = 5):
    behavior = behavior.squeeze(1).numpy()
    num = len(behavior)
    cum_hist, _ = np.histogram(behavior, np.linspace(obj_lb, obj_ub, n_bins + 1))
    cum_hist = cum_hist[np.nonzero(cum_hist)] / (num) # discrete distribution
    cum_hist_uni = np.mean(cum_hist) * np.ones_like(cum_hist) # theoretical uniform distribution

    cum_coverage = len(cum_hist) / n_bins
    cum_uniformity = 1 - jensenshannon(cum_hist, cum_hist_uni, base=2)
    
    return cum_coverage, cum_uniformity
    

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
    k = 10
    obj_lb = 0
   
    obj_ub = 1 # maximum for 6D Hartmann
    
    cost_tensor = []
    coverage_tensor = []
    uniformity_tensor = []
    cumbent_tensor = []
    
    for seed in range(replicate):
        print('seed:',seed)
        np.random.seed(seed)
        
        sobol = SobolEngine(dimension=dim, scramble=True, seed=seed)
        train_x = sobol.draw(n=N_init).to(torch.float64)
        
        train_y = []
        for k in range(N_init):
            fun = Maze()
            train_y.append(fun.run_experiment(lb+(ub-lb)*train_x[k].unsqueeze(0)))
         
        train_y = torch.tensor(train_y).unsqueeze(1)
    
        coverage, uniformity = reachability_uniformity(train_y, n_bins, obj_lb, obj_ub) # Calculate the initial reachability and uniformity
        
        coverage_list = [coverage]
        uniformity_list = [uniformity]
        cost_list = [0] # number of sampled data excluding initial data
        cumbent_list = [float(max(train_y))]
        
        # Start BO loop
        for i in range(BO_iter):        
            
           
            
            sobol = SobolEngine(dimension=dim, scramble=True, seed=seed)
            train_x = sobol.draw(n=int(N_init+i+1)).to(torch.float64)
            fun = Maze()
            Y_next = fun.run_experiment(lb+(ub-lb)*train_x[-1])
            train_y = torch.cat((train_y, Y_next))
            
            
            coverage, uniformity = reachability_uniformity(train_y, n_bins, obj_lb, obj_ub)
            coverage_list.append(coverage)
            uniformity_list.append(uniformity)
            cost_list.append(cost_list[-1]+1)
            cumbent_list.append(float(max(train_y)))
            
        cost_tensor.append(cost_list)
        cumbent_tensor.append(cumbent_list)
        coverage_tensor.append(coverage_list)
        uniformity_tensor.append(uniformity_list)
        torch.save(train_x, 'Maze_train_x_sobol_seed'+str(seed)+'.pt')
        torch.save(train_y, 'Maze_train_y_sobol_seed'+str(seed)+'.pt')
    
    cumbent_tensor = torch.tensor(cumbent_tensor, dtype=torch.float32) 
    cost_tensor = torch.tensor(cost_tensor, dtype=torch.float32) 
    coverage_tensor = torch.tensor(coverage_tensor, dtype=torch.float32) 
    uniformity_tensor = torch.tensor(uniformity_tensor, dtype=torch.float32)  
    torch.save(coverage_tensor, 'MediumMaze_coverage_list_sobol.pt')
    # torch.save(uniformity_tensor, 'MediumMaze_uniformity_list_sobol.pt')
    torch.save(cost_tensor, 'MediumMaze_cost_list_sobol.pt')  
    torch.save(cumbent_tensor, 'MediumMaze_cumbent_list_sobol.pt')  
