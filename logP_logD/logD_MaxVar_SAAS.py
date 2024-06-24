#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 13 17:11:29 2024

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

def reachability_uniformity(behavior, n_bins = 25, obj_lb = -5, obj_ub = 5, n_hist = 25):
    behavior = behavior.squeeze(1).numpy()
    num = len(behavior)
    cum_hist, _ = np.histogram(behavior, np.linspace(obj_lb, obj_ub, n_bins + 1))
    cum_hist = cum_hist[np.nonzero(cum_hist)] / (num) # discrete distribution
    cum_hist_uni = np.mean(cum_hist) * np.ones_like(cum_hist) # theoretical uniform distribution

    cum_coverage = len(cum_hist) / n_hist
    cum_uniformity = 1 - jensenshannon(cum_hist, cum_hist_uni, base=2)
    
    return cum_coverage, cum_uniformity
    
    
class CustomAcquisitionFunction():
    def __init__(self, model, sampled_behavior, k=10):
        
        self.model = model
        self.k = k
        self.sampled_behavior = sampled_behavior
       
    
    def __call__(self, X):
        """Compute the acquisition function value at X."""
        
        
        # Obtain model predictions
        posterior = self.model.posterior(X)
        acquisition_values = posterior.mixture_variance
        #(posterior is on original scale) https://github.com/pytorch/botorch/blob/main/botorch/models/gp_regression.py               
        # acquisition_values = posterior.variance.mean(dim=0)
               
        return acquisition_values.flatten()
    

if __name__ == '__main__':
    device = torch.device("cpu")
    dim = 125
    N_init = 10
    replicate = 20
    n_bins = 25 # number of bins used to calculate the reachability and uniformity 
    k = 10 # number of k-nearest neighbor
    BO_iter = 100
    warmup_steps = 128
    num_samples = 64
    thinning = 16
    
    cost_tensor = []
    coverage_tensor = []
    
    # Data pre-processing
    #####################################################################################################################################################
    path = os.getcwd()
    data_path = path+"/Training Data/extract_data_logD.csv"
    col_list=['LogD','Exp_RT']
    lc_df = pd.read_csv(data_path,usecols=col_list)

    # Remove non_retained molecules
    index=lc_df[lc_df['Exp_RT'] < 180].index
    lc_df.drop(lc_df[lc_df['Exp_RT'] < 180].index,inplace=True)

    # Import descriptor file
    path = os.getcwd()
    data_path = path+"/Training Data/descriptors_logD.csv"
    des_df = pd.read_csv(data_path,index_col=0)

    # Remove non_retained molecules
    des_df  = des_df.drop(des_df.index[index]) # modified from original code
    
    data_set_1 = des_df
    data_set_2 = lc_df
    des_with_lc = pd.concat([data_set_1,data_set_2],axis=1)
    des_with_lc_feat_corr = des_with_lc.columns[des_with_lc.corrwith(des_with_lc['LogD']) >=0.90][:-1]
    des_with_lc = des_with_lc.drop(columns=des_with_lc_feat_corr)

    # Filling the nan with mean values in des_with_lc
    for col in des_with_lc:
        des_with_lc[col].fillna(des_with_lc[col].mean(),inplace=True)

    # Remove columns with zero vlues
    des_with_lc = des_with_lc.loc[:,(des_with_lc**2).sum() != 0]
    data = des_with_lc.drop(['LogD'],axis=1)

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
    y_original = des_with_lc['LogD'].values
    ##############################################################################################################################################
    
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
        ids_acquired = np.random.choice(np.arange((len(y_original))), size=N_init, replace=False)
        train_x = X_original[ids_acquired]
        train_y = y_original[ids_acquired] # original scale

        coverage, uniformity = reachability_uniformity(train_y, n_bins, obj_lb, obj_ub, n_hist) # Calculate the initial reachability and uniformity
        
        coverage_list = [coverage]
        cost_list = [0] # number of sampled data excluding initial data
        
        # Start BO loop
        for i in range(BO_iter):        
                        
            #SAASBO
            model = SaasFullyBayesianSingleTaskGP(train_X=train_x.to(device), train_Y=train_y.to(device), outcome_transform=Standardize(m=1))
            fit_fully_bayesian_model_nuts(model, warmup_steps=warmup_steps, num_samples=num_samples, thinning=thinning)
            
            custom_acq_function = CustomAcquisitionFunction(model, train_y, k=k)
            
            # Optimize the acquisition function (discrete)
            acquisition_values = custom_acq_function(X_original)
            ids_sorted_by_aquisition = acquisition_values.argsort(descending=True)
            for id_max_aquisition_all in ids_sorted_by_aquisition: # we can't sample point that is already sampled
                if not id_max_aquisition_all.item() in ids_acquired:
                    id_max_aquisition = id_max_aquisition_all.item()
                    break
                
            ids_acquired = np.concatenate((ids_acquired, [id_max_aquisition]))
            train_x = X_original[ids_acquired]
            train_y = y_original[ids_acquired] # original scale
            
            
            coverage, uniformity = reachability_uniformity(train_y, n_bins, obj_lb, obj_ub, n_hist)
            coverage_list.append(coverage)
            cost_list.append(cost_list[-1] + len([id_max_aquisition]))
    
       
        cost_tensor.append(cost_list)
        coverage_tensor.append(coverage_list)
       
    
    cost_tensor = torch.tensor(cost_tensor, dtype=torch.float32) 
    coverage_tensor = torch.tensor(coverage_tensor, dtype=torch.float32) 
    
    torch.save(coverage_tensor, 'logD_coverage_list_MaxVar_SAAS.pt')
    torch.save(cost_tensor, 'logD_cost_list_MaxVar_SAAS.pt')  
    


