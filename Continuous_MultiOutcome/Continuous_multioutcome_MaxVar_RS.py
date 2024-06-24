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

def synthetic_function(x):

    y1 = torch.sin(x[:,0]) * torch.cos(x[:,1]) + x[:,2] * torch.exp(-x[:,0]**2) * torch.cos(x[:,0] + x[:,1]) + 0.01*np.sin((x[:,3] + x[:,4] + x[:,5]))
    y2 = torch.sin(x[:,3]) * torch.cos(x[:,4]) + x[:,5] * torch.exp(-x[:,3]**2) * torch.sin(x[:,3] + x[:,4]) + 0.01*np.cos((x[:,0] + x[:,1] + x[:,2]))

    return torch.cat((y1.unsqueeze(1),y2.unsqueeze(1)),dim=1)
    
def reachability_uniformity(behavior, n_bins = 25, obj_lb1 = -5, obj_ub1 = 5, obj_lb2 = -5, obj_ub2 = 5):
    
    cell_size1 = (obj_ub1 - obj_lb1) / n_bins
    cell_size2 = (obj_ub2 - obj_lb2) / n_bins
    
    indices_y1 = ((behavior[:, 0] - obj_lb1) / cell_size1).floor().int()
    indices_y2 = ((behavior[:, 1] - obj_lb2) / cell_size2).floor().int()
    grid_indices = torch.stack((indices_y1, indices_y2), dim=1)
    unique_cells = set(map(tuple, grid_indices.tolist()))
    return len(unique_cells)/(n_bins**2)

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
    
    dim = 6
    N_init = 10
    BO_iter = 500
    lb = -5
    ub = 5
    n_bins = 10
    obj_lb1 = -5.1
    obj_ub1 = 5.1
    obj_lb2 = -5.1
    obj_ub2 = 5.1
    
    replicate = 20
    RandomSearch = False
       
    cost_list = [[] for _ in range(replicate)]
    coverage_list = [[] for _ in range(replicate)]
   
    
    for seed in range(replicate):
        print('seed:',seed)
        np.random.seed(seed)
        
        train_x = torch.tensor(np.random.rand(N_init, dim))
        train_y = synthetic_function(lb+(ub-lb)*train_x)
        
        cost_list[seed].append(0)
        coverage = reachability_uniformity(train_y, n_bins, obj_lb1, obj_ub1, obj_lb2, obj_ub2)
        coverage_list[seed].append(coverage)
        
        for bo_iter in range(BO_iter):
            model_list = []
            for nx in range(2):
                covar_module = ScaleKernel(RBFKernel(ard_num_dims=dim))
                model_list.append(SingleTaskGP(train_x.to(torch.float64), train_y[:,nx].unsqueeze(1).to(torch.float64), outcome_transform=Standardize(m=1), covar_module=covar_module))
            model = ModelListGP(*model_list)
            mll = SumMarginalLogLikelihood(model.likelihood, model)
            try:
                fit_gpytorch_mll(mll)
            except:
                print('Cant fit GP!')
            
            acquisition_function = MaxVariance(model)
                      
            bounds = torch.tensor([[0.0]*dim, [1.0]*dim], dtype=torch.float)
                        
            if RandomSearch:
                candidate = torch.rand(1,dim)
            else:
                candidate, acq = optimize_acqf(
                    acquisition_function, bounds=bounds, q=1, num_restarts=10, raw_samples=20,
                )
                
            train_x = torch.cat((train_x, candidate))
            Y_next = synthetic_function(lb+(ub-lb)*candidate)
            train_y = torch.cat((train_y, Y_next))
            
            coverage = reachability_uniformity(train_y, n_bins, obj_lb1, obj_ub1, obj_lb2, obj_ub2)
            coverage_list[seed].append(coverage)
            cost_list[seed].append(cost_list[seed][-1]+1)
            

    cost_tensor = torch.tensor(cost_list, dtype=torch.float32) 
    coverage_tensor = torch.tensor(coverage_list, dtype=torch.float32) 
    
    torch.save(coverage_tensor, 'Cluster_coverage_list_MaxVar.pt')
    torch.save(cost_tensor, 'Cluster_cost_list_MaxVar.pt')  
