#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  5 19:37:06 2024

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
from botorch.acquisition import ExpectedImprovement
from botorch.optim import optimize_acqf
import cma


dim = 8
replicate = 10
N_init = 20
lb = -5
ub = 5
BO_iter = 200
fun = Ackley(dim = dim)
reachability_list = [[] for _ in range(replicate)]
cost_list = [[] for _ in range(replicate)]
for seed in range(replicate):
    np.random.seed(seed)
    
    train_X = torch.tensor(np.random.rand(N_init, dim))
    train_Y = fun(lb + (ub-lb)*train_X)
    cost_list[seed].append(0)
    reachability_list[seed].append(float(train_Y.var()))
    
    x0 = train_X[torch.argmin(train_Y)].numpy()
    sigma0 = 1.0
    es = cma.CMAEvolutionStrategy(x0, sigma0)
    while not es.stop():
        solutions = es.ask()
        es.tell(solutions, [float(fun(torch.tensor(x))) for x in solutions])
        train_Y = torch.cat((train_Y, fun(torch.tensor(solutions))))
        es.logger.add()
        es.disp()
        
        cost_list[seed].append(len(train_Y))
        reachability_list[seed].append(float(train_Y.var()))
        
        if len(train_Y)>BO_iter-1:
            break
        
torch.save(torch.tensor(cost_list), '8DAckley_cost_list_CMAES.pt')     
torch.save(torch.tensor(reachability_list), '8DAckley_variance_list_CMAES.pt')   


