#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run the synthetic continuous multi-outcome BEACON study.

Inputs: synthetic two-output function defined in this script.
Outputs: coverage.pt in results/generated/continuous/multi-outcome/synthetic-beacon.
Runtime: expensive; intended for reproducing a paper experiment, not a smoke test.
"""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

import torch
from botorch.models import SingleTaskGP, ModelListGP
from gpytorch.kernels import MaternKernel, RBFKernel, ScaleKernel
from botorch.utils.transforms import t_batch_mode_transform
from botorch.acquisition import AcquisitionFunction, AnalyticAcquisitionFunction
import numpy as np
from botorch.models.transforms.outcome import Standardize
from gpytorch.mlls.sum_marginal_log_likelihood import SumMarginalLogLikelihood
from beacon.optimization import DTYPE, fit_gpytorch_mll_quiet, optimize_acqf_quiet
from beacon.paths import GENERATED_RESULTS_DIR
from beacon.thompson_sampling import EfficientThompsonSampler

def synthetic_function(x):
    '''Synthetic multi-output function studied in the paper'''
    y1 = torch.sin(x[:,0]) * torch.cos(x[:,1]) + x[:,2] * torch.exp(-x[:,0]**2) * torch.cos(x[:,0] + x[:,1]) + 0.01*np.sin((x[:,3] + x[:,4] + x[:,5]))
    y2 = torch.sin(x[:,3]) * torch.cos(x[:,4]) + x[:,5] * torch.exp(-x[:,3]**2) * torch.sin(x[:,3] + x[:,4]) + 0.01*np.cos((x[:,0] + x[:,1] + x[:,2]))
    
    return torch.cat((y1.unsqueeze(1),y2.unsqueeze(1)),dim=1)

def reachability(behavior, n_bins, obj_lb, obj_ub):
    '''Compute the reachability for given behaviors.'''
    
    indices_list = []
    for n_output in range(N_output):
        cell_size = (obj_ub[n_output] - obj_lb[n_output]) / n_bins
        indices_list.append(((behavior[:, n_output] - obj_lb[n_output]) / cell_size).floor().int())
    
    grid_indices = torch.stack(indices_list,1)
    unique_cells = set(map(tuple, grid_indices.tolist()))
    return len(unique_cells)/(n_bins**2)

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
            self.ts_sampler_list[n_output].calculate_V()
        
    @t_batch_mode_transform(expected_q=1)
    def forward(self, X):
        """Compute the acquisition function value at X."""
        
        sample_list = []
        for n_output in range(N_output):
            sample_list.append(self.ts_sampler_list[n_output].query_sample(X)) # Sample from posterior
        
        samples = torch.cat(sample_list,1)
       
        samples = samples.to(dtype=self.sampled_behavior.dtype, device=self.sampled_behavior.device)
        dist = torch.cdist(samples, self.sampled_behavior) # calculate the two norm between each Thompson sampled behavior with sampled behavior
        dist, _ = torch.sort(dist, dim = 1) # sort the distance 
        n = dist.size()[1]
        nearest = min(self.k_NN, n)
        E = torch.cat(
            (
                torch.ones(nearest, dtype=dist.dtype, device=dist.device),
                torch.zeros(n - nearest, dtype=dist.dtype, device=dist.device),
            ),
            dim=0,
        ) # find the k-nearest neighbor
        dist = dist*E
        acquisition_values = torch.sum(dist, dim=1)
        
        return acquisition_values.flatten()

if __name__ == '__main__':
       
    dim = 6 # dimension for the problem
    N_init = 10 # number of initial training data point for GP
    replicate = 20 # number of replicate for experiment
    BO_iter = 500 # number of BO iteration
    n_bins = 10 # number of bins for dividing the outcome space
    N_output = 2 # number of outcome 
    k_NN = 10 # number of k-nearest value
    obj_lb = [-5.1, -5.1] # minimum value for each outcome
    obj_ub = [5.1, 5.1] # maximum value for each outcome
    lb = -5 # lower bound for feature space
    ub = 5 # upper bound for feature space
    
    cost_tensor = []
    coverage_tensor = []
    
    for seed in range(replicate):
        print('seed:',seed)
        np.random.seed(seed)
        
        # generate initial training data
        train_x = torch.tensor(np.random.rand(N_init, dim), dtype=DTYPE)
        train_y = synthetic_function(lb+(ub-lb)*train_x).to(dtype=DTYPE)

        coverage = reachability(train_y, n_bins, obj_lb, obj_ub) # Calculate the initial reachability        
        coverage_list = [coverage]
       
        # Start BO loop
        for i in range(BO_iter):    
            print('Iteration:', i)
            
            # build model list
            model_list = []
            for nx in range(N_output):
                covar_module = ScaleKernel(RBFKernel(ard_num_dims=dim)).to(dtype=DTYPE)
                model_list.append(SingleTaskGP(train_x, train_y[:,nx].unsqueeze(1), outcome_transform=Standardize(m=1), covar_module=covar_module))
            model = ModelListGP(*model_list)
            mll = SumMarginalLogLikelihood(model.likelihood, model)
            
            # Fit the GPs
            try:
                fit_gpytorch_mll_quiet(mll, model=model, seed=seed, iteration=i)
            except Exception as exc:
                raise RuntimeError(f"Failed to fit GP at seed={seed}, iteration={i}") from exc
            
            for nx in range(N_output):
                model.models[nx].train_x = train_x
                model.models[nx].train_y = train_y[:,nx].unsqueeze(1)
           
            custom_acq_function = CustomAcquisitionFunction(model, train_y, k_NN=k_NN) # define our custom acquisition function               
            bounds = torch.tensor([[0.0]*dim, [1.0]*dim], dtype=DTYPE) # define the bound for optimizing acquisition function
            
            # Optimize the acquisition function
            candidate, acq_value = optimize_acqf_quiet(
                acq_function=custom_acq_function,
                bounds=bounds,
                q=1,
                seed=seed,
                iteration=i,
                model=model,
            )
             
            train_x = torch.cat((train_x, candidate)) # append the new query point
            y_new = synthetic_function(lb+(ub-lb)*candidate).to(dtype=DTYPE) # new query function value
            train_y = torch.cat((train_y, y_new)) # append the new query function value
                       
            coverage = reachability(train_y, n_bins, obj_lb, obj_ub) # calculate the new reachability
            coverage_list.append(coverage)
                       
        coverage_tensor.append(coverage_list)

    coverage_tensor = torch.tensor(coverage_tensor, dtype=torch.float32)

    output_dir = GENERATED_RESULTS_DIR / "continuous" / "multi-outcome" / Path(__file__).stem
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(coverage_tensor, output_dir / "coverage.pt")
    print(f"Saved generated results to {output_dir}")
