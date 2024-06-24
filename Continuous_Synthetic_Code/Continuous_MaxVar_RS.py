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
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from botorch.utils import standardize
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.optim import optimize_acqf
import numpy as np
from scipy.spatial.distance import cdist, jensenshannon
from botorch.models.transforms.outcome import Standardize
from botorch.test_functions import Rosenbrock, Ackley, Hartmann, StyblinskiTang
import gpytorch
import time
 
def reachability_uniformity(behavior, n_bins = 25, obj_lb = -5, obj_ub = 5):
    behavior = behavior.squeeze(1).numpy()
    num = len(behavior)
    cum_hist, _ = np.histogram(behavior, np.linspace(obj_lb, obj_ub, n_bins + 1))
    cum_hist = cum_hist[np.nonzero(cum_hist)] / (num) # discrete distribution
    cum_hist_uni = np.mean(cum_hist) * np.ones_like(cum_hist) # theoretical uniform distribution

    cum_coverage = len(cum_hist) / n_bins
    cum_uniformity = 1 - jensenshannon(cum_hist, cum_hist_uni, base=2)
    
    return cum_coverage, cum_uniformity

class MaxVariance(AnalyticAcquisitionFunction):
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
        super().__init__(model=model, posterior_transform=posterior_transform)
        self.maximize = maximize

    @t_batch_mode_transform(expected_q=1)
    def forward(self, X: Tensor) -> Tensor:
        r"""Evaluate the posterior variance on the candidate set X.

        Args:
            X: A `(b1 x ... bk) x 1 x d`-dim batched tensor of `d`-dim design points.

        Returns:
            A `(b1 x ... bk)`-dim tensor of posterior variance values at the
            given design points `X`.
        """
        mean, sigma = self._mean_and_sigma(X)
        
        return sigma**2

if __name__ == '__main__':    
    
    dim = 20
    N_init = 300
    BO_iter = 1000
    lb = -5
    ub = 5
    n_bins = 100
   
    # Specify the minimum/maximum value for each synthetic function as to calculate reachability
    
    obj_lb = 0 # minimum obj value for Rosenbrock
    # obj_ub = 270108 # obj maximum for 4D Rosenbrock
    # obj_ub = 630252.63 # obj maximum for 8D Rosenbrock
    # obj_ub = 990396.990397 # obj maximum for 12D Rosenbrock
    obj_ub = 1710685.71
    
    # obj_lb = 0 # minimum obj value for Ackley
    # obj_ub = 14.3027 # maximum obj value for Ackley
    
    # obj_lb = -39.16599*dim # minimum obj val for 4D SkyTang
    # obj_ub = 500 # maximum obj val for 4D SkyTang
    # obj_ub = 1000 # maximum obj val for 8D SkyTang
    # obj_ub = 1500 # maximum obj for 12D SkyTang
   
    # Specify the synthetic function we want to study
    
    fun = Rosenbrock(dim=dim)
    # fun = Ackley(dim=dim)
    # fun = StyblinskiTang(dim=dim)
    
    replicate = 5
    RandomSearch =True# Random search or MaxVar
      
    cost_list = [[] for _ in range(replicate)]
    coverage_list = [[] for _ in range(replicate)]
    time_tensor = []
    for seed in range(replicate):
        start_time = time.time()
        print('seed:',seed)
        np.random.seed(seed)
        
        train_X = torch.tensor(np.random.rand(N_init, dim))
        train_Y = fun(lb + (ub-lb)*train_X).unsqueeze(1)
        
        cost_list[seed].append(0)
        coverage, uniformity = reachability_uniformity(train_Y, n_bins, obj_lb, obj_ub)
        coverage_list[seed].append(coverage)
        
        for bo_iter in range(BO_iter):
            covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(ard_num_dims=dim)) # Choose the RBF kernel
            gp = SingleTaskGP(train_X, train_Y, covar_module = covar_module, outcome_transform=Standardize(m=1))
            mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
            # fit_gpytorch_mll(mll)
            
            acquisition_function = MaxVariance(gp)
                      
            bounds = torch.tensor([[0.0]*dim, [1.0]*dim], dtype=torch.float)
                        
            if RandomSearch:
                candidate = torch.rand(1,dim)
            else:
                candidate, acq = optimize_acqf(
                    acquisition_function, bounds=bounds, q=1, num_restarts=10, raw_samples=20,
                )
                
            train_X = torch.cat((train_X, candidate))
            Y_next = fun(lb+(ub-lb)*candidate).unsqueeze(1)
            train_Y = torch.cat((train_Y, Y_next))
            
            coverage, uniformity = reachability_uniformity(train_Y, n_bins, obj_lb, obj_ub)
            coverage_list[seed].append(coverage)
            cost_list[seed].append(cost_list[seed][-1]+1)
            
        end_time = time.time()
        time_tensor.append((end_time-start_time)/BO_iter)
        
    time_tensor = torch.tensor(time_tensor, dtype=torch.float32) 
    cost_tensor = torch.tensor(cost_list, dtype=torch.float32) 
    coverage_tensor = torch.tensor(coverage_list, dtype=torch.float32) 
    torch.save(coverage_tensor, '20DRosen_coverage_list_RS.pt')
    torch.save(cost_tensor, '20DRosen_cost_list_RS.pt')  
    torch.save(time_tensor, '20DRosen_time_list_RS.pt')
