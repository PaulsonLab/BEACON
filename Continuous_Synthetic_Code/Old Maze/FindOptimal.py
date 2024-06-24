#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 20 00:45:47 2024

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
dim = 54
N_init = 100000
lb = -100
ub = 100


sobol = SobolEngine(dimension=dim, scramble=True)
train_x =sobol.draw(n=N_init).to(torch.float64)
# torch.manual_seed(seed)
# train_x = torch.rand(N_init,dim)
# train_x = (train_x - train_x.min(dim=0).values)/(train_x.max(dim=0).values-train_x.min(dim=0).values)

# train_y = function((lb+(ub-lb)*train_x)).unsqueeze(1)
train_y = []
# for k in range(N_init):
#     # train_x.append(torch.tensor(np.random.rand(1, dim)))
#     train_y.append(Maze(train_x[k].unsqueeze(0).to(torch.float32)))
for k in range(N_init):
    fun = Maze()
    train_y.append(fun.run_experiment(lb+(ub-lb)*train_x[k].unsqueeze(0)))
    best_y = float(max(train_y))
    # if train_y[-1]>0.99:
    #     break
# train_x = torch.stack(train_x).squeeze(1)
# train_y = torch.tensor(train_y).unsqueeze(1)
torch.save(train_x,'X_optimal_hard.pt')
torch.save(train_y,'Y_optimal_hard.pt')