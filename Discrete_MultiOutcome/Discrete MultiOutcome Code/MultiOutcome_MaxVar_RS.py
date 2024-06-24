#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  3 13:25:37 2024

@author: tang.1856
"""

from botorch.acquisition import AcquisitionFunction, AnalyticAcquisitionFunction
import torch
from botorch.models.model import Model
from typing import Dict, Optional, Tuple, Union
from torch import Tensor
from botorch.acquisition.objective import PosteriorTransform
from botorch.utils.transforms import convert_to_target_pre_hook, t_batch_mode_transform
from botorch.models import SingleTaskGP, SaasFullyBayesianSingleTaskGP, ModelListGP
from botorch.fit import fit_gpytorch_mll
from botorch.utils import standardize
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.optim import optimize_acqf
import numpy as np
from scipy.spatial.distance import cdist, jensenshannon
from botorch.models.transforms.outcome import Standardize
from botorch.test_functions import Rosenbrock, Ackley, Hartmann, StyblinskiTang
import gpytorch
from gpytorch.mlls.sum_marginal_log_likelihood import SumMarginalLogLikelihood
from gpytorch.kernels import MaternKernel, RBFKernel, ScaleKernel
import matplotlib.pyplot as plt
import gymnasium as gym
import pandas as pd
    
def reachability_uniformity(behavior, n_bins = 25, mins = None, maxs=None, num_filled_grids = 100):

    filled_grids = set()
    grid_sizes = (maxs - mins) / (n_bins)
    # Assign points to grids
    for point in behavior:
        if all(point>=maxs):
            grid_indices = torch.tensor([n_bins-1,n_bins-1])
        else:
            grid_indices = ((point - mins) / grid_sizes).int()
        # Use a tuple (immutable) to store grid indices in the set
        filled_grids.add(tuple(grid_indices.tolist()))
        
        
    return len(filled_grids)/(num_filled_grids)


class MaxVariance(AcquisitionFunction):
    r"""Single-outcome Max Variance Acq.

    Example:
        >>> model = SingleTaskGP(train_X, train_Y)
        >>> MaxVar = MaxVariance(model)
        >>> maxvar = MaxVar(test_X)
    """

    def __init__(
        self,
        model: Model,
        posterior_transform: Optional[PosteriorTransform] = None,
        maximize: bool = True,
    ) -> None:
        r"""Single-outcome Max Variance.

        Args:
            model: A fitted single-outcome GP model (must be in batch mode if
                candidate sets X will be)
            posterior_transform: A PosteriorTransform. If using a multi-output model,
                a PosteriorTransform that transforms the multi-output posterior into a
                single-output posterior is required.
            maximize: If True, consider the problem a maximization problem.
        """
        super().__init__(model=model)
        

    @t_batch_mode_transform(expected_q=1)
    def forward(self, X: Tensor) -> Tensor:
        r"""Evaluate the posterior variance on the candidate set X.

        Args:
            X: A `(b1 x ... bk) x 1 x d`-dim batched tensor of `d`-dim design points.

        Returns:
            A `(b1 x ... bk)`-dim tensor of posterior variance values at the
            given design points `X`.
        """
      
        posterior1 = self.model.models[0].posterior(X=X)
        posterior2 = self.model.models[1].posterior(X=X)
        variance1 = posterior1.variance.clamp_min(1e-12)
        variance2 = posterior2.variance.clamp_min(1e-12)
        variance = variance1 + variance2
        return variance.flatten()

if __name__ == '__main__':    
    
    dim = 25
    N_init = 50
    BO_iter = 500
    
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
    n_bins = 10
    
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
            grid_indices = torch.tensor([n_bins-1,n_bins-1])
        else:
            grid_indices = ((point - mins) / grid_sizes).int()
        # Use a tuple (immutable) to store grid indices in the set
        filled_grids.add(tuple(grid_indices.tolist()))
    
    # Number of unique grids that contain at least one data point
    num_filled_grids = len(filled_grids)
    
    replicate = 20
    RandomSearch = True
    
    
    cost_list = [[] for _ in range(replicate)]
    coverage_list = [[] for _ in range(replicate)]
   
    for seed in range(replicate):
        print('seed:',seed)
        np.random.seed(seed)
        ids_acquired = np.random.choice(np.arange((len(y1))), size=N_init, replace=False)
       
        train_x = X_original[ids_acquired]
        train_y = Y_original[ids_acquired]
        
        
        cost_list[seed].append(0)
        coverage = reachability_uniformity(train_y, n_bins, mins,maxs, num_filled_grids)
        coverage_list[seed].append(coverage)

        
        for bo_iter in range(BO_iter):
            model_list = []
            for nx in range(2):
                covar_module = ScaleKernel(RBFKernel(ard_num_dims=dim))
                model_list.append(SingleTaskGP(train_x.to(torch.float64), train_y[:,nx].unsqueeze(1).to(torch.float64), outcome_transform=Standardize(m=1), covar_module=covar_module))
            model = ModelListGP(*model_list)
            mll = SumMarginalLogLikelihood(model.likelihood, model)
            # try:
            #     fit_gpytorch_mll(mll)
            # except:
            #     print('Cant fit GP!')
            
         
            if RandomSearch:
                
                acquisition_values = torch.rand(len(X_original))
            else:
                acquisition_values = (model.models[0].posterior(X_original).variance + model.models[1].posterior(X_original).variance).flatten()
                
            ids_sorted_by_aquisition = acquisition_values.argsort(descending=True)
            for id_max_aquisition_all in ids_sorted_by_aquisition:
                if not id_max_aquisition_all.item() in ids_acquired:
                    id_max_aquisition = id_max_aquisition_all.item()
                    break
    
          
            ids_acquired = np.concatenate((ids_acquired, [id_max_aquisition]))    
            
            train_x = X_original[ids_acquired]            
            train_y = Y_original[ids_acquired]
            
            coverage = reachability_uniformity(train_y, n_bins, mins,maxs, num_filled_grids)
            coverage_list[seed].append(coverage)
            cost_list[seed].append(cost_list[seed][-1]+1)
        
    cost_tensor = torch.tensor(cost_list, dtype=torch.float32) 
    coverage_tensor = torch.tensor(coverage_list, dtype=torch.float32) 
    torch.save(coverage_tensor, 'MOF_coverage_list_RS.pt')
    torch.save(cost_tensor, 'MOF_cost_list_RS.pt')  
