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
from botorch.acquisition import UpperConfidenceBound, ExpectedImprovement
import sys
sys.path.append('/home/tang.1856/Jonathan/Hands-on-Neuroevolution-with-Python-master/Chapter6')
from maze_NS import Maze
from torch.quasirandom import SobolEngine
from gpytorch.kernels import MaternKernel, RBFKernel, ScaleKernel
class toy_function():
    def __init__(self, dim):
        self.dim = dim
        
    def __call__(self, x):
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
        return sigma

if __name__ == '__main__':    
    
    dim = 22
    N_init = 30
    BO_iter = 300
    # lb = -5
    # ub = 5
    n_bins = 25
    obj_lb = 0
    obj_ub = 1
    
    replicate = 20
    RandomSearch =False
    EI =True
    
    MaxVar =False
    UCB = False

    
    cost_list = [[] for _ in range(replicate)]
    coverage_list = [[] for _ in range(replicate)]
    uniformity_list = [[] for _ in range(replicate)]
    cumbent_list = [[] for _ in range(replicate)]
    
    for seed in range(replicate):
        
        print('seed:',seed)
        np.random.seed(seed)
        
        X_opt = -30+60*torch.tensor([[0.0029, 0.2438, 0.3485, 0.5100, 0.6259, 0.1782, 0.1210, 0.1420, 0.9787,
                0.6091, 0.7387, 0.4671, 0.8452, 0.5360, 0.4084, 0.0163, 0.3617, 0.4038,
                0.3541, 0.2491, 0.8930, 0.6666]])
        
        lb = X_opt-7
        ub = X_opt+7
        
       
        sobol = SobolEngine(dimension=dim, scramble=True,seed=seed)
        train_X =sobol.draw(n=N_init).to(torch.float64)
        
        train_Y = []
        # for k in range(N_init):
        #     # train_x.append(torch.tensor(np.random.rand(1, dim)))
        #     train_y.append(Maze(train_x[k].unsqueeze(0).to(torch.float32)))
        for k in range(N_init):
            fun = Maze()
            train_Y.append(fun.run_experiment(lb+(ub-lb)*train_X[k].unsqueeze(0))[0])
            # if train_y[-1]>0.95:
            #     break
        # train_x = torch.stack(train_x).squeeze(1)
        train_Y = torch.tensor(train_Y).unsqueeze(1)
        
        # train_X = torch.tensor((np.random.rand(N_init, dim)-0.5-lb)/(ub-lb)) # generate the same intial data as the evolutionay-based NS
        # train_X = torch.tensor(np.random.rand(N_init, dim))
        # train_Y = fun(lb + (ub-lb)*train_X).unsqueeze(1)
        cumbent_list[seed].append(float(max(train_Y)))
        cost_list[seed].append(0)
        coverage, uniformity = reachability_uniformity(train_Y, n_bins, obj_lb, obj_ub)
        coverage_list[seed].append(coverage)
        uniformity_list[seed].append(uniformity)
        for bo_iter in range(BO_iter):
            covar_module = ScaleKernel(RBFKernel(ard_num_dims=dim))
            gp = SingleTaskGP(train_X.to(torch.float64), train_Y.to(torch.float64), outcome_transform=Standardize(m=1), covar_module=covar_module)
            mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
            fit_gpytorch_mll(mll)
            # gp.covar_module.base_kernel.lengthscale = 0.3 # Setting the lengthscale
            # gp.covar_module.outputscale = 1.0  # Setting the outputscale 
            
                      
            bounds = torch.tensor([[0.0]*dim, [1.0]*dim], dtype=torch.float)
                        
            if RandomSearch:
                
                candidate = torch.rand(1,dim)
            elif EI:
                acquisition_function = ExpectedImprovement(gp, best_f=float(max(train_Y)))
                candidate, acq = optimize_acqf(
                    acquisition_function, bounds=bounds, q=1, num_restarts=10, raw_samples=20,
                )
                
            elif MaxVar:
                acquisition_function = MaxVariance(gp)
                candidate, acq = optimize_acqf(
                    acquisition_function, bounds=bounds, q=1, num_restarts=10, raw_samples=20,
                )
            elif UCB:
                acquisition_function = UpperConfidenceBound(gp, beta=4)
                candidate, acq = optimize_acqf(
                    acquisition_function, bounds=bounds, q=1, num_restarts=10, raw_samples=20,
                )
                
            train_X = torch.cat((train_X, candidate))
            fun = Maze()
            Y_next,_,_ = fun.run_experiment(lb+(ub-lb)*candidate)
            train_Y = torch.cat((train_Y, Y_next))
            
            coverage, uniformity = reachability_uniformity(train_Y, n_bins, obj_lb, obj_ub)
            coverage_list[seed].append(coverage)
            uniformity_list[seed].append(uniformity)
            cost_list[seed].append(cost_list[seed][-1]+1)
            cumbent_list[seed].append(float(max(train_Y)))
            torch.save(train_X, 'Maze_MaxVar_train_x_seed'+str(seed)+'.pt')
            torch.save(train_Y, 'Maze_MaxVar_train_y_seed'+str(seed)+'.pt')

    cost_tensor = torch.tensor(cost_list, dtype=torch.float32) 
    coverage_tensor = torch.tensor(coverage_list, dtype=torch.float32) 
    uniformity_tensor = torch.tensor(uniformity_list, dtype=torch.float32)  
    cumbent_tensor = torch.tensor(cumbent_list, dtype=torch.float32) 

    torch.save(coverage_tensor, 'MediumMaze_coverage_list_EI_.pt')
    # torch.save(uniformity_tensor, 'MediumMaze_uniformity_list_MaxVar_.pt')
    torch.save(cost_tensor, 'MediumMaze_cost_list_EI_.pt')  
    torch.save(cumbent_tensor, 'MediumMaze_cumbent_list_EI_.pt')  

    
    
