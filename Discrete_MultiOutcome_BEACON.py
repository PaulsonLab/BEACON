#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 10 13:26:09 2024

@author: tang.1856
"""
import torch
from botorch.models import SingleTaskGP, SaasFullyBayesianSingleTaskGP, ModelListGP
from gpytorch.kernels import MaternKernel, RBFKernel, ScaleKernel
from torch.quasirandom import SobolEngine
from botorch.fit import fit_gpytorch_mll
from botorch.utils.transforms import t_batch_mode_transform
from botorch.acquisition import AcquisitionFunction, AnalyticAcquisitionFunction
from scipy.spatial.distance import cdist, jensenshannon
import numpy as np
from botorch.models.transforms.outcome import Standardize
from gpytorch.mlls.sum_marginal_log_likelihood import SumMarginalLogLikelihood
from ThompsonSampling import EfficientThompsonSampler
import pandas as pd

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
    
    # joint gas uptake capacity case study
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
    feat = list(feat)
    file_path1 = '/home/tang.1856/BEACON/BEACON/Material Data/PMOF20K_traindata_7000_train.csv'
    data1 = pd.read_csv(file_path1)
    y1 = data1['pure_uptake_CO2_298.00_15000']
    y2 = data1['pure_uptake_methane_298.00_580000']
    
    lb = torch.tensor(data1[feat].values.min(axis=0))
    ub = torch.tensor(data1[feat].values.max(axis=0))
    X_original = (torch.tensor(data1[feat].values) - lb)/(ub-lb)
    Y_original = torch.stack((torch.tensor(y1.values),torch.tensor(y2.values)),dim=1)
        
    dim = 25 # dimension of feature space
    N_init = 50 # number of initial training data for GP
    replicate = 20
    BO_iter = 500
    n_bins = 10
    TS = 1 
    k_NN = 10
    N_output = 2
        
    coverage_tensor = []
        
    for seed in range(replicate):
        
        print('seed:',seed)
        
        np.random.seed(seed)
        ids_acquired = np.random.choice(np.arange((len(y1))), size=N_init, replace=False)
       
        train_x = X_original[ids_acquired]
        train_y = Y_original[ids_acquired]
        
        # Start BEACON loop
        for i in range(BO_iter):        
            print('Iteration:', i)
                        
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
                       