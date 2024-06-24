#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  6 16:26:33 2024

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

def reachability_uniformity(behavior, n_bins = 25, obj_lb = -5, obj_ub = 5):
    behavior = behavior.squeeze(1).numpy()
    num = len(behavior)
    cum_hist, _ = np.histogram(behavior, np.linspace(obj_lb, obj_ub, n_bins + 1))
    cum_hist = cum_hist[np.nonzero(cum_hist)] / (num) # discrete distribution
    cum_hist_uni = np.mean(cum_hist) * np.ones_like(cum_hist) # theoretical uniform distribution

    cum_coverage = len(cum_hist) / n_bins
    cum_uniformity = 1 - jensenshannon(cum_hist, cum_hist_uni, base=2)
    
    return cum_coverage, cum_uniformity
    
        
class CustomAcquisitionFunction_Mean():
    def __init__(self, model, sampled_behavior, k=10, TS = 1):
        
        self.model = model
        self.k = k
        self.sampled_behavior = sampled_behavior
        self.TS = TS
    
    def __call__(self, X):
        """Compute the acquisition function value at X."""
        
        # Obtain model predictions
        posterior = self.model.posterior(X)
        # samples = posterior.rsample(sample_shape=torch.Size([1])) # Thompson Sampling (posterior is on original scale) https://github.com/pytorch/botorch/blob/main/botorch/models/gp_regression.py
        samples = posterior.mean.unsqueeze(0)
        dist = torch.norm(samples - self.sampled_behavior, dim=0)
        dist, _ = torch.sort(dist, dim = 1) # sort the distance 
        n = dist.size()[1]

        E = torch.cat((torch.ones(self.k), torch.zeros(n-self.k)), dim = 0) # find the k-nearest neighbor
        dist = dist*E.unsqueeze(0).unsqueeze(-1)
        acquisition_values = torch.sum(dist,dim=1)
        
        # For H2 data set
        # posterior = self.model.posterior(X[:,0:1000])
        # samples = posterior.rsample(sample_shape=torch.Size([self.TS])) # Thompson Sampling
        # for r in range(1,math.ceil(len(X[0])/1000)):           
        #     posterior = self.model.posterior(X[:, int(r*1000):int((r+1))*1000])
        #     samples_new = posterior.rsample(sample_shape=torch.Size([self.TS])) # Thompson Sampling
        #     samples = torch.cat((samples, samples_new), dim=2)
        
        # posterior = self.model.posterior(X)
        # # samples = posterior.rsample(sample_shape=torch.Size([self.TS])) # Thompson Sampling
        # samples = posterior.mean
        # sampled_behavior_expanded = self.sampled_behavior.reshape(-1).unsqueeze(0)
        # acquisition_values = []
        # for ts in range(self.TS): # different TS sample
        #     samples_expanded = samples[ts, 0, :, :].expand(-1, self.sampled_behavior.size(0)) # ts thompsonsampling points
        #     dist = abs(samples_expanded - sampled_behavior_expanded) # calculate two norm wrt outcome space
        #     dist, _ = torch.sort(dist, dim = 1) # sort the distance 
        #     n = dist.size()[1]
        #     E = torch.cat((torch.ones(self.k), torch.zeros(n-self.k)), dim = 0) # find the k-nearest neighbor
        #     dist = dist*E
        #     acquisition_values.append(torch.sum(dist,dim=1))
            
        # acquisition_values = torch.stack(acquisition_values)
        # acquisition_values = torch.max(acquisition_values, dim=0).values
        
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
    
    dim = 7
    N_init = 10
    replicate = 20
    n_bins = 25 # number of bins used to calculate the reachability and uniformity 
    k = 10 # number of k-nearest neighbor
    TS = 5 # number of TS samples
    BO_iter = 80
    
    cost_tensor = []
    coverage_tensor = []
    uniformity_tensor = []
    rmse_list = []
    cumbent_list = [[] for _ in range(replicate)]
    
    # load H2 adsorption MOF data from pkl file 
    X_original = pickle.load(open('Training Data/hydrogen_input_output.pkl', 'rb'))['x'] # data from Ghude and Chowdhury 2023 (7 features for MOFs)
    y_original = pickle.load(open('Training Data/hydrogen_input_output.pkl', 'rb'))['y'] # data from Ghude and Chowdhury 2023 (H2 adsorp capacity)
    
    # load SourGas MOF data from xlsx file
    # df = pd.read_excel('/home/tang.1856/Jonathan/Novelty Search/Training Data/SourGas.xlsx') # data from Cho et al.
    # X_original = (df.iloc[:, 1:(1+dim)]).values
    # y_original = df.iloc[:, (1+dim)].values
    
    # load xylene separation data from xlsx file 
    # df = pd.read_excel('/home/tang.1856/Jonathan/Novelty Search/Training Data/xylene.xlsx') # data from Qiao et al.
    # X_original = (df.iloc[:, 2:(2+dim)]).values
    # y_original = df.iloc[:, (2+dim)].values
    
    # load solubility data from csv file 
    # df = pd.read_csv('/home/tang.1856/Jonathan/Novelty Search/Training Data/water_set_wide_descriptors.csv') # data from Boobier et al.
    # X_original = (df.iloc[:, 4:(4+dim)]).values
    # y_original = df.iloc[:, 3].values
    
    
    
    y_original = np.reshape(y_original, (np.size(y_original), 1)) 
    X_original = torch.from_numpy(X_original)
    
    X_original = (X_original - X_original.min(dim=0).values)/(X_original.max(dim=0).values - X_original.min(dim=0).values) # normalize the original input data
    y_original = torch.from_numpy(y_original)
    
    obj_lb = y_original.min() # obj minimum
    obj_ub = y_original.max() # obj maximum 
    
    for seed in range(replicate):
        print('seed:',seed)
        np.random.seed(seed)
        ids_acquired = np.random.choice(np.arange((len(y_original))), size=N_init, replace=False)
        train_x = X_original[ids_acquired]
        train_y = y_original[ids_acquired] # original scale
       
        coverage, uniformity = reachability_uniformity(train_y, n_bins, obj_lb, obj_ub) # Calculate the initial reachability and uniformity
        
        coverage_list = [coverage]
        uniformity_list = [uniformity]
        cost_list = [0] # number of sampled data excluding initial data
        
        cumbent_list[seed].append(max(train_y))
        # Start BO loop
        for i in range(BO_iter):        
            
            # covar_module = ScaleKernel(base_kernel)
            model = SingleTaskGP(train_x, train_y, outcome_transform=Standardize(m=1))
            mll = ExactMarginalLogLikelihood(model.likelihood, model)
            fit_gpytorch_mll(mll)
            
            custom_acq_function = CustomAcquisitionFunction_Mean(model, train_y, k=k, TS = TS)
            
            # Optimize the acquisition function (discrete)
            acquisition_values = custom_acq_function(X_original.unsqueeze(1))
            ids_sorted_by_aquisition = acquisition_values.argsort(descending=True)
            for id_max_aquisition_all in ids_sorted_by_aquisition:
                if not id_max_aquisition_all.item() in ids_acquired:
                    id_max_aquisition = id_max_aquisition_all.item()
                    break
                
            ids_acquired = np.concatenate((ids_acquired, [id_max_aquisition]))
            train_x = X_original[ids_acquired]
            train_y = y_original[ids_acquired] # original scale
            
            cumbent_list[seed].append(max(train_y))
            
            
            coverage, uniformity = reachability_uniformity(train_y, n_bins, obj_lb, obj_ub)
            coverage_list.append(coverage)
            uniformity_list.append(uniformity)
            cost_list.append(cost_list[-1] + len([id_max_aquisition]))
    
        rmse = calculate_rmse(X_original, y_original, train_x, train_y, ids_acquired) # use the sampled data to fit GP and calculate the rnse for unseen data
        rmse_list.append(rmse)
        cost_tensor.append(cost_list)
        coverage_tensor.append(coverage_list)
        uniformity_tensor.append(uniformity_list)
    
    cost_tensor = torch.tensor(cost_tensor, dtype=torch.float32) 
    coverage_tensor = torch.tensor(coverage_tensor, dtype=torch.float32) 
    uniformity_tensor = torch.tensor(uniformity_tensor, dtype=torch.float32)  
    cumbent_tensor = torch.tensor(cumbent_list, dtype=torch.float32)
    torch.save(coverage_tensor, 'H2_coverage_list_NS_mean.pt')
    torch.save(uniformity_tensor, 'H2_uniformity_list_NS_mean.pt')
    torch.save(cost_tensor, 'H2_cost_list_NS_mean.pt')  
    # torch.save(cumbent_tensor, 'H2_cumbent_list_NS.pt')
    
    print('Avg RMSE = ', sum(rmse_list)/len(rmse_list))
    


