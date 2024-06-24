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
import os
from botorch.models.fully_bayesian import SaasFullyBayesianSingleTaskGP
from botorch import fit_fully_bayesian_model_nuts

def toy_function(x):
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
    
# class CustomAcquisitionFunction(AcquisitionFunction):
#     def __init__(self, model, sampled_behavior, k=10):
#         super().__init__(model=model)
#         self.model = model
#         self.k = k
#         self.sampled_behavior = sampled_behavior
#         # self.y_std = y_std
#         # self.y_mean = y_mean
        
#     @t_batch_mode_transform(expected_q=1)
#     def forward(self, X):
#         """Compute the acquisition function value at X."""
#         # Ensure X is at least 2D
#         if X.dim() == 1:
#             X = X.unsqueeze(0)
        
#         # Obtain model predictions
#         posterior = self.model.posterior(X)
#         # samples = posterior.rsample(sample_shape=torch.Size([1])) # Thompson Sampling (posterior is on original scale) https://github.com/pytorch/botorch/blob/main/botorch/models/gp_regression.py
#         samples = posterior.rsample(sample_shape=torch.Size([1])).mean(dim=2)
#         # samples = self.y_mean + samples*self.y_std # Rescale
#         dist = torch.norm(samples - self.sampled_behavior, dim=0)
#         dist, _ = torch.sort(dist, dim = 1) # sort the distance 
#         n = dist.size()[1]

#         E = torch.cat((torch.ones(self.k), torch.zeros(n-self.k)), dim = 0) # find the k-nearest neighbor
#         dist = dist*E.unsqueeze(0).unsqueeze(-1)
#         acquisition_values = torch.sum(dist,dim=1)
        
#         return acquisition_values.flatten()
    
class CustomAcquisitionFunction():
    def __init__(self, model, sampled_behavior, k=10):
        
        self.model = model
        self.k = k
        self.sampled_behavior = sampled_behavior
       
    
    def __call__(self, X):
        """Compute the acquisition function value at X."""
        
        # Obtain model predictions
        posterior = self.model.posterior(X)
        samples = posterior.rsample(sample_shape=torch.Size([1])).squeeze(1) # Thompson Sampling (posterior is on original scale) https://github.com/pytorch/botorch/blob/main/botorch/models/gp_regression.py
        # samples = posterior.rsample(sample_shape=torch.Size([1])).mean(dim=2) # SaasBO
        dist = torch.cdist(samples[0].to(torch.float64), self.sampled_behavior.to(torch.float64)).squeeze(0)
        # dist = torch.norm(samples - self.sampled_behavior, dim=0)
        dist, _ = torch.sort(dist, dim = 1) # sort the distance 
        n = dist.size()[1]

        E = torch.cat((torch.ones(self.k), torch.zeros(n-self.k)), dim = 0) # find the k-nearest neighbor
        dist = dist*E
        acquisition_values = torch.sum(dist,dim=1)
        
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
    fit_gpytorch_model(mll)
    
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
    
    dim = 125
    N_init = 10
    replicate = 20
    n_bins = 25 # number of bins used to calculate the reachability and uniformity 
    k = 10 # number of k-nearest neighbor
    BO_iter = 100

    cost_tensor = []
    coverage_tensor = []
    uniformity_tensor = []
    rmse_list = []
    cumbent_list = [[] for _ in range(replicate)]
    
    # Data pre-processing
    #####################################################################################################################################################
    path = os.getcwd()
    data_path = path+"/Training Data/extract_data_logP.csv"
    col_list=['LogP','Exp_RT']
    lc_df = pd.read_csv(data_path,usecols=col_list)

    # Remove non_retained molecules
    index=lc_df[lc_df['Exp_RT'] < 180].index
    lc_df.drop(lc_df[lc_df['Exp_RT'] < 180].index,inplace=True)

    # Import descriptor file
    path = os.getcwd()
    data_path = path+"/Training Data/descriptors_logP.csv"
    des_df = pd.read_csv(data_path,index_col=0)

    # Remove non_retained molecules
    des_df  = des_df.drop(des_df.index[index]) # modified from original code
    
    data_set_1 = des_df
    data_set_2 = lc_df
    des_with_lc = pd.concat([data_set_1,data_set_2],axis=1)
    des_with_lc_feat_corr = des_with_lc.columns[des_with_lc.corrwith(des_with_lc['LogP']) >=0.90][:-1]
    des_with_lc = des_with_lc.drop(columns=des_with_lc_feat_corr)

    # Filling the nan with mean values in des_with_lc
    for col in des_with_lc:
        des_with_lc[col].fillna(des_with_lc[col].mean(),inplace=True)

    # Remove columns with zero vlues
    des_with_lc = des_with_lc.loc[:,(des_with_lc**2).sum() != 0]
    data = des_with_lc.drop(['LogP'],axis=1)

    # Remove features with low Variance(threshold<=0.05)
    data_var = data.var()
    del_feat = list(data_var[data_var <= 0.05].index)
    data.drop(columns=del_feat, inplace=True)

    # Remove features with correlation(threshold > 0.95)
    corr_matrix = data.corr().abs()
    mask = np.triu(np.ones_like(corr_matrix,dtype=bool))
    tri_df = corr_matrix.mask(mask)
    to_drop =  [ c for c in tri_df.columns if any(tri_df[c] > 0.95)]
    data = data.drop(to_drop,axis=1)
    X_original = (data.iloc[:, 1:(1+dim)]).values
    y_original = des_with_lc['LogP'].values
    ##############################################################################################################################################
    
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
       
        # y_std = train_y.std()
        # y_mean = train_y.mean()
        # Y = standardize(train_y) # standardize
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
            
            custom_acq_function = CustomAcquisitionFunction(model, train_y, k=k)
            
            # Optimize the acquisition function (discrete)
            acquisition_values = custom_acq_function(X_original.unsqueeze(0))
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
    torch.save(coverage_tensor, 'logP_TS_1_coverage_list_NS.pt')
    torch.save(uniformity_tensor, 'logP_TS_1_uniformity_list_NS.pt')
    torch.save(cost_tensor, 'logP_TS_1_cost_list_NS.pt')  
    # torch.save(cumbent_tensor, 'H2_cumbent_list_NS.pt')
    
    print('Avg RMSE = ', sum(rmse_list)/len(rmse_list))
    


