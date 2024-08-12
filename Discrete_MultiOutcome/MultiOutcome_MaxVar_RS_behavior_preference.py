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
# import gymnasium as gym
import pandas as pd
from OilSorbent import OilSorbent
from pyDOE2 import lhs
import itertools
    
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
    return reachability, filled


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
    BO_iter = 200
    replicate = 20
    
    # Oil sorbent data set
    fun = OilSorbent(dim)
    lb = torch.tensor([3/7, 0.7, 12, 0.12, 0, 16, 0.41])
    ub = torch.tensor([7/3, 2, 24, 0.18, 0.2, 28, 1.32])
    rand_samp = lhs(dim, 2, random_state=0) # using latin hypercube
    X_original = torch.tensor(rand_samp, dtype=torch.double)
    Y_original = fun(lb+(ub-lb)*X_original)
    Y_original[Y_original<0] = 0
    
    n_bins = 10
    
    
    
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
    
    
    RandomSearch = True
    
    
    cost_list = [[] for _ in range(replicate)]
    coverage_list = [[] for _ in range(replicate)]
    coverage_original,_ = reachability_uniformity(xl,xu,Y_original)
    for seed in range(replicate):
        print('seed:',seed)
        np.random.seed(seed)
        ids_acquired = np.random.choice(np.arange((len(Y_original))), size=N_init, replace=False)
       
        train_x = X_original[ids_acquired]
        train_y = Y_original[ids_acquired]
        
        
        cost_list[seed].append(0)
        coverage, filled_total = reachability_uniformity(xl,xu,train_y)
        coverage /=coverage_original
        coverage_list[seed].append(coverage)

        
        for bo_iter in range(BO_iter):
            model_list = []
            for nx in range(2):
                covar_module = ScaleKernel(RBFKernel(ard_num_dims=dim))
                model_list.append(SingleTaskGP(train_x.to(torch.float64), train_y[:,nx].unsqueeze(1).to(torch.float64), outcome_transform=Standardize(m=1), covar_module=covar_module))
            model = ModelListGP(*model_list)
            mll = SumMarginalLogLikelihood(model.likelihood, model)
            
            if not RandomSearch:
                try:
                    fit_gpytorch_mll(mll)
                except:
                    print('Cant fit GP!')
            
         
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
            
            coverage, filled_total = reachability_uniformity(xl,xu,train_y)
            coverage /=coverage_original
            coverage_list[seed].append(coverage)
            cost_list[seed].append(cost_list[seed][-1]+1)
        
    cost_tensor = torch.tensor(cost_list, dtype=torch.float32) 
    coverage_tensor = torch.tensor(coverage_list, dtype=torch.float32) 
    torch.save(coverage_tensor, 'Oil_coverage_list_RS_120.pt')
    torch.save(cost_tensor, 'Oil_cost_list_RS_120.pt')  
