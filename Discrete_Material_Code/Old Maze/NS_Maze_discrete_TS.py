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
# from test_functions import push_function
import pickle
from botorch.models.transforms.outcome import Standardize
from botorch import fit_gpytorch_model
import matplotlib.pyplot as plt
import pandas as pd
import math
import sys
sys.path.append('/home/tang.1856/Jonathan/Hands-on-Neuroevolution-with-Python-master/Chapter6')
from maze_NS import Maze
import visualize

def reachability_uniformity(behavior, n_bins = 25, obj_lb = -5, obj_ub = 5):
    behavior = behavior.squeeze(1).numpy()
    num = len(behavior)
    cum_hist, _ = np.histogram(behavior, np.linspace(obj_lb, obj_ub, n_bins + 1))
    cum_hist = cum_hist[np.nonzero(cum_hist)] / (num) # discrete distribution
    cum_hist_uni = np.mean(cum_hist) * np.ones_like(cum_hist) # theoretical uniform distribution

    cum_coverage = len(cum_hist) / n_bins
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
            X: A `batch_shape x N x d`-dim Tensor from which to sample (in the `N`
                dimension)
            num_samples: The number of samples to draw.
            
        """
        
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
        # samples = posterior.rsample(sample_shape=torch.Size([self.TS])).squeeze(2)
        
        acquisition_values = []
        for ts in range(self.TS): # different TS sample
            
            dist = torch.cdist(samples[ts].to(torch.float32), self.sampled_behavior.to(torch.float32)).squeeze(0)
            dist, _ = torch.sort(dist, dim = 1) # sort the distance 
            n = dist.size()[1]
            E = torch.cat((torch.ones(self.k), torch.zeros(n-self.k)), dim = 0) # find the k-nearest neighbor
            dist = dist*E
            acquisition_values.append(torch.sum(dist,dim=1))
            
        acquisition_values = torch.stack(acquisition_values)
        acquisition_values = torch.max(acquisition_values, dim=0).values
        
        
        return acquisition_values.flatten()
    
def calculate_rmse(X, y, X_BO, Y_BO, ids_acquired):
    # ids_acquired_test = np.random.choice(np.arange((len(y))), size=len(y), replace=False) # sample all data from the original data set
    mask = np.ones(len(X),dtype=bool)
    mask[ids_acquired]=False
    X_test = X[mask]
    Y_test = y[mask].numpy()
    # Y_test = y[ids_acquired_test].numpy()
    # X_test = X[ids_acquired_test]
    
    model = SingleTaskGP(X_BO, Y_BO, outcome_transform=Standardize(m=1)) # X_BO and Y_BO are the sampled data for BO
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_mll(mll)
    
    y_pred = model.posterior(X_test).mean.detach().numpy()
    rmse = np.sqrt(np.mean((Y_test - y_pred)**2))
    
    # Try plotting parity plot using the data we sampled 
    # plt.scatter(Y_test, y_pred)
    # # plt.plot([min(y), max(y)], [min(y), max(y)], 'k--', label='Ideal Fit') 
    # plt.xlabel('True value')
    # plt.ylabel('predicted value')
    # plt.title('Parity Plot (MaxVar)')
    # plt.grid(True)
    return rmse
    
if __name__ == '__main__':
    
    dim = 22
    N_init = 10
    N_init_original = 20000
    replicate = 20
    n_bins = 25 # number of bins used to calculate the reachability and uniformity 
    k = 10 # number of k-nearest neighbor
    TS = 1 # number of TS samples
    BO_iter = 300
    
    cost_tensor = []
    coverage_tensor = []
    uniformity_tensor = []
    rmse_list = []
    cumbent_list = [[] for _ in range(replicate)]
    
    # sobol = SobolEngine(dimension=dim, scramble=True,seed=0)
    # # X_original = sobol.draw(n=N_init).to(torch.float64)
    # X_original = [torch.tensor([[0.0029, 0.2438, 0.3485, 0.5100, 0.6259, 0.1782, 0.1210, 0.1420, 0.9787,
    #         0.6091, 0.7387, 0.4671, 0.8452, 0.5360, 0.4084, 0.0163, 0.3617, 0.4038,
    #         0.3541, 0.2491, 0.8930, 0.6666]])]
    # X_perturbation = -0.25+0.5*sobol.draw(n=N_init_original).to(torch.float64)
    # fun = Maze()
    # y_original = [fun.run_experiment(X_original[0].unsqueeze(0))]
    # for q in range(N_init_original-1):
    #     X_original.append(X_perturbation[q]+X_original[0])
    #     fun = Maze()
    #     y_original.append(fun.run_experiment(X_original[q+1].unsqueeze(0)))
    # y_original = torch.tensor(y_original).unsqueeze(1)
    # X_original = torch.stack(X_original).squeeze(1)  
    # torch.save(X_original,'X_original.pt')
    # torch.save(y_original,'y_original.pt')
    
    X_original = torch.load('X_original.pt')
    y_original = torch.load('y_original.pt')
    # y_original = np.reshape(y_original, (np.size(y_original), 1)) 
    # X_original = torch.from_numpy(X_original)
    
    
    # y_original = torch.from_numpy(y_original)
    
    obj_lb = 0 # obj minimum
    obj_ub = 1 # obj maximum 
    
    # fun = Maze()
    # _, path_points = fun.run_experiment(X_original[0].unsqueeze(0))
    # fun = Maze()
    # visualize.draw_agent_path(fun.maze_env, path_points)
    for seed in range(replicate):
        
        np.random.seed(seed)
        ids_acquired = np.random.choice(np.arange((len(y_original))), size=5000, replace=False)
        X_original = X_original[ids_acquired]
        y_original = y_original[ids_acquired]
        
        X_original_nor = (X_original - X_original.min(dim=0).values)/(X_original.max(dim=0).values - X_original.min(dim=0).values) # normalize the original input data
        
        print('seed:',seed)
        np.random.seed(seed)
        ids_acquired = np.random.choice(np.arange((len(y_original))), size=N_init, replace=False)
        train_x = X_original_nor[ids_acquired]
        train_y = y_original[ids_acquired] # original scale
       
        coverage, uniformity = reachability_uniformity(train_y, n_bins, obj_lb, obj_ub) # Calculate the initial reachability and uniformity
        
        coverage_list = [coverage]
        uniformity_list = [uniformity]
        cost_list = [0] # number of sampled data excluding initial data
        
        cumbent_list[seed].append(float(max(train_y)))
        # Start BO loop
        for i in range(BO_iter):        
            
            # covar_module = ScaleKernel(base_kernel)
            model = SingleTaskGP(train_x.to(torch.float64), train_y.to(torch.float64), outcome_transform=Standardize(m=1))
            mll = ExactMarginalLogLikelihood(model.likelihood, model)
            fit_gpytorch_mll(mll)
            
            # Set the lengthscale and outputscale
            # model.covar_module.base_kernel.lengthscale = 0.5 # Setting the lengthscale
            # model.covar_module.outputscale = 1.0  # Setting the outputscale 
            
            custom_acq_function = CustomAcquisitionFunction_TS(model, train_y, k=k, TS = TS)
            
            # Optimize the acquisition function (discrete)
            acquisition_values = custom_acq_function(X_original_nor)
            ids_sorted_by_aquisition = acquisition_values.argsort(descending=True)
            for id_max_aquisition_all in ids_sorted_by_aquisition:
                if not id_max_aquisition_all.item() in ids_acquired:
                    id_max_aquisition = id_max_aquisition_all.item()
                    break
                
            ids_acquired = np.concatenate((ids_acquired, [id_max_aquisition]))
            train_x = X_original_nor[ids_acquired]
            train_y = y_original[ids_acquired] # original scale
            
            cumbent_list[seed].append(float(max(train_y)))
            
            
            coverage, uniformity = reachability_uniformity(train_y, n_bins, obj_lb, obj_ub)
            coverage_list.append(coverage)
            uniformity_list.append(uniformity)
            cost_list.append(cost_list[-1] + len([id_max_aquisition]))
    
        # rmse = calculate_rmse(X_original, y_original, train_x, train_y, ids_acquired) # use the sampled data to fit GP and calculate the rnse for unseen data
        # rmse_list.append(rmse)
        cost_tensor.append(cost_list)
        coverage_tensor.append(coverage_list)
        uniformity_tensor.append(uniformity_list)
    
    cost_tensor = torch.tensor(cost_tensor, dtype=torch.float32) 
    coverage_tensor = torch.tensor(coverage_tensor, dtype=torch.float32) 
    uniformity_tensor = torch.tensor(uniformity_tensor, dtype=torch.float32)  
    cumbent_tensor = torch.tensor(cumbent_list, dtype=torch.float32)
    torch.save(coverage_tensor, 'MediumMaze_TS_1_coverage_list_NS.pt')
    torch.save(uniformity_tensor, 'MediumMaze_TS_1_uniformity_list_NS.pt')
    torch.save(cost_tensor, 'MediumMaze_TS_1_cost_list_NS.pt')  
    # torch.save(cumbent_tensor, 'H2_cumbent_list_NS.pt')
    
    # print('Avg RMSE = ', sum(rmse_list)/len(rmse_list))
    


