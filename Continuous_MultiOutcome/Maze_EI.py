#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  3 12:04:22 2024

@author: tang.1856
"""

import gymnasium as gym
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
from botorch.acquisition import AcquisitionFunction, AnalyticAcquisitionFunction
from botorch.optim import optimize_acqf
from scipy.spatial.distance import cdist, jensenshannon
import numpy as np
from torch.quasirandom import SobolEngine
from botorch.test_functions import Rosenbrock, Ackley, Hartmann, StyblinskiTang
import torchsort
import pickle
from botorch.models.transforms.outcome import Standardize
import matplotlib.pyplot as plt
from gpytorch.kernels import RBFKernel, ScaleKernel
# from ThompsonSampling import EfficientThompsonSampler
import tqdm
from botorch.models.model import Model
from typing import Dict, Optional, Tuple, Union
from torch import Tensor
from botorch.acquisition.objective import PosteriorTransform
from botorch.acquisition import ExpectedImprovement

def policy(param, state):
       
    p1 = param[0]*state[0] + param[1]*state[1] + param[2]*state[2] + param[3]*state[3]
    p1 = -1+2*torch.sigmoid(p1)
    
    p2 = param[4]*state[0] + param[5]*state[1] + param[6]*state[2] + param[7]*state[3]
    p2 = -1+2*torch.sigmoid(p2)
    
    return [float(p1),float(p2)]
    
def environment(param):    
    
    env = gym.make("PointMaze_Large-v3", continuing_task=False)
    options = {'goal_cell':np.array([5,2]), 'reset_cell':np.array([7,4])}
    observation, info = env.reset(seed=10, options=options)
    initial_dist = np.linalg.norm(observation['achieved_goal']-observation['desired_goal'])
    
    for i in range(300):

       action = policy(param, observation['observation'])
       observation, reward, terminated, truncated, info = env.step(action)

       if terminated or truncated:
          observation['achieved_goal'] = observation['desired_goal']         
          break
    
    
    final_dist = np.linalg.norm(observation['achieved_goal']-observation['desired_goal'])
    Reward = (initial_dist-final_dist)/initial_dist
    env.close()

    return Reward



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
       
        posterior = self.model.posterior(X=X)
        
        variance = posterior.variance.clamp_min(1e-12)
                
        return variance.flatten()

if __name__ == '__main__':    
    
    dim = 8
    lb = -1
    ub = 1
    
    N_init = 20
    BO_iter = 300           
    replicate = 20
           
    cost_list = [[] for _ in range(replicate)]
    coverage_list = [[] for _ in range(replicate)]
    cumbent_list = [[] for _ in range(replicate)]
    
    for seed in range(replicate):
        print('seed:',seed)
        np.random.seed(seed)
               
        train_X = torch.tensor(np.random.rand(N_init+10, dim))
        train_x, train_y= [], []
        for element in train_X:
            reward = environment(lb+(ub-lb)*element)      
            if reward<0.9:
                train_x.append(element.tolist())
                train_y.append(reward)
        train_x = torch.tensor(train_x)
        train_y = torch.tensor(train_y).unsqueeze(1)
        cost_list[seed].append(0)
        cumbent_list[seed].append(float(max(train_y)))
              
        for bo_iter in range(BO_iter):
            
            covar_module = ScaleKernel(RBFKernel(ard_num_dims=dim))
            model = SingleTaskGP(train_x.to(torch.float64), train_y.to(torch.float64), outcome_transform=Standardize(m=1), covar_module=covar_module)           
            mll = ExactMarginalLogLikelihood(model.likelihood, model)
           
            try:
                fit_gpytorch_mll(mll)
            except:
                print('Cant fit GP!')
            
            bounds = torch.tensor([[0.0]*dim, [1.0]*dim], dtype=torch.float)
                        
            acquisition_function = ExpectedImprovement(model, best_f=max(train_y))
            candidate, acq = optimize_acqf(
                acquisition_function, bounds=bounds, q=1, num_restarts=10, raw_samples=20,
            )
           
                
            train_x = torch.cat((train_x, candidate))
            Y_next = torch.tensor(environment(lb+(ub-lb)*candidate[0])).unsqueeze(0).unsqueeze(1)       
            train_y = torch.cat((train_y, Y_next))
            
            cost_list[seed].append(cost_list[seed][-1]+1)
            cumbent_list[seed].append(float(max(train_y)))
            
        # torch.save(train_x, 'Maze_EI_train_x_seed'+str(seed)+'.pt')
        
    cost_tensor = torch.tensor(cost_list, dtype=torch.float32) 
    cumbent_tensor = torch.tensor(cumbent_list, dtype=torch.float32) 
    torch.save(cumbent_tensor, 'Maze_cumbent_list_EI.pt')
    torch.save(cost_tensor, 'Maze_cost_list_EI.pt')
    
