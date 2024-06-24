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

def policy(param, state):
    p1 = param[0]*state[0] + param[1]*state[1] + param[2]*state[2] + param[3]*state[3] + param[4]
    # p1 = (torch.sigmoid(p1) >=0.5)*1
    p1 = (p1>0.0)*1
    return int(p1)
    
def environment(param):    
    
    env = gym.make("CartPole-v1")
    observation, info = env.reset(seed=10)
    reward_acc = 0
    for i in range(500):
       # action = env.action_space.sample()  # this is where you would insert your policy
       # param = torch.rand(1,dim).flatten()
       action = policy(param, observation)
       # print(action)
       observation, reward, terminated, truncated, info = env.step(action)
       reward_acc+=1
       if terminated or truncated:
          # observation, info = env.reset()
          print('reward=',i) 
          
          break
    
    env.close()
    return observation, reward_acc

dim = 5
lb = -1.0
ub = 1.0
# lb = torch.tensor([0.1,0.1,0.25,0.15,0])
# ub = torch.tensor([0.2,0.2,0.35,0.25,0.1])

N_init = 100
train_x = torch.rand(N_init,dim)

train_y1, train_y2 = [], []
for element in train_x:
    obs, reward = environment(lb+(ub-lb)*element)
    # if reward>100:
    train_y1.append(obs[0])
    train_y2.append(obs[0])
    if reward>=499:
        print(element)

train_y = torch.tensor(train_y2).unsqueeze(1)
covar_module = ScaleKernel(RBFKernel(ard_num_dims=dim)) # select the RBF kernel
model = SingleTaskGP(train_x, train_y, outcome_transform=Standardize(m=1), covar_module=covar_module)
mll = ExactMarginalLogLikelihood(model.likelihood, model)
fit_gpytorch_mll(mll)

test_x = torch.rand(N_init,dim)
test_y1, test_y2 = [], []
for element in test_x:
    obs, reward = environment(lb+(ub-lb)*element)
    # if reward>100:
    test_y1.append(obs[0])
    test_y2.append(obs[0])
posterior = model.posterior(X = test_x)
pred = posterior.mean

plt.scatter(test_y2, pred.detach().numpy())

