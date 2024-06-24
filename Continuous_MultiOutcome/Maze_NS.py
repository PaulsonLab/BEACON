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

def policy(param, state):
    # p1 = param[0]*state[0] + param[1]*state[1] + param[2]*state[2] + param[3]*state[3] + param[4]
    # p1 = -1+2*torch.sigmoid(p1)
    
    # p2 = param[5]*state[0] + param[6]*state[1] + param[7]*state[2] + param[8]*state[3] + param[9]
    # p2 = -1+2*torch.sigmoid(p2)
    
    p1 = param[0]*state[0] + param[1]*state[1] + param[2]*state[2] + param[3]*state[3]
    p1 = -1+2*torch.sigmoid(p1)
    
    p2 = param[4]*state[0] + param[5]*state[1] + param[6]*state[2] + param[7]*state[3]
    p2 = -1+2*torch.sigmoid(p2)
    
    return [float(p1),float(p2)]
    
def environment(param):    
    
    env = gym.make("PointMaze_Large-v3", continuing_task=False)
    options = {'goal_cell':np.array([5,2]), 'reset_cell':np.array([7,4])}
    observation, info = env.reset(seed=10, options=options)
    
    reward_acc = 0
    for i in range(300):
       
       # action = env.action_space.sample()  # this is where you would insert your policy
       # param = torch.rand(1,dim).flatten()
       action = policy(param, observation['observation'])
       # print(action)
       observation, reward, terminated, truncated, info = env.step(action)
       # reward_acc+=reward
       if terminated or truncated:
          observation['achieved_goal'] = observation['desired_goal']
          # observation, info = env.reset()
          # print('reward=',reward_acc) 
          
          break
    print('reward=',reward)
    # if reward_acc>0.0:
    #     print(param)
    env.close()
    return observation['achieved_goal'], reward_acc

dim = 8
lb = -1
ub = 1
# lb = torch.tensor([-0.9828, -0.7703, -0.2976, -0.8634, -0.2421,  0.4134, -0.9801, -0.3291, -0.0069, -0.4247])-0.5
# ub = torch.tensor([-0.9828, -0.7703, -0.2976, -0.8634, -0.2421,  0.4134, -0.9801, -0.3291, -0.0069, -0.4247])+0.5
# lb = torch.tensor([-1,-1,-1,-1,-1,0,-1,-1,-1,-1])
# ub = torch.tensor([0,0,0,0,0,1,0,0,0,0])
# lb = torch.tensor([-1,-1,0,-1,0,-1,-1,0])
# ub = torch.tensor([0,0,1,0,1,0,0,1])
N_init = 500
train_x = torch.rand(N_init,dim)

train_y1, train_y2 = [], []
for element in train_x:
    obs, reward = environment(lb+(ub-lb)*element)
    # if reward>100:
    train_y1.append(obs[0])
    train_y2.append(obs[1])
    # if reward>=499:
    #     print(element)

train_y = torch.tensor(train_y2).unsqueeze(1)
# covar_module = ScaleKernel(MaternKernel(ard_num_dims=dim, nu=1.5)) # select the RBF kernel
covar_module = ScaleKernel(RBFKernel(ard_num_dims=dim))
model = SingleTaskGP(train_x.to(torch.float64), train_y, outcome_transform=Standardize(m=1), covar_module=covar_module)
mll = ExactMarginalLogLikelihood(model.likelihood, model)
fit_gpytorch_mll(mll)


test_x = torch.rand(100,dim)
test_y1, test_y2 = [], []
for element in test_x:
    obs, reward = environment(lb+(ub-lb)*element)
    # if reward>100:
    test_y1.append(obs[0])
    test_y2.append(obs[1])
posterior = model.posterior(X = test_x)
pred = posterior.mean

# plt.scatter(test_y2, pred.detach().numpy())


# DKL
train_x = train_x.cuda()
train_y = train_y.flatten().cuda()
data_dim = train_x.size(-1)

class LargeFeatureExtractor(torch.nn.Sequential):
    def __init__(self):
        super(LargeFeatureExtractor, self).__init__()
        self.add_module('linear1', torch.nn.Linear(data_dim, 100))
        self.add_module('relu1', torch.nn.ReLU())
        self.add_module('linear2', torch.nn.Linear(100, 50))
        self.add_module('relu2', torch.nn.ReLU())
        self.add_module('linear3', torch.nn.Linear(50, 10))
        self.add_module('relu3', torch.nn.ReLU())
        self.add_module('linear4', torch.nn.Linear(10, 2))

feature_extractor = LargeFeatureExtractor()

class GPRegressionModel(gpytorch.models.ExactGP):
        def __init__(self, train_x, train_y, likelihood):
            super(GPRegressionModel, self).__init__(train_x, train_y, likelihood)
            self.mean_module = gpytorch.means.ConstantMean()
            self.covar_module = gpytorch.kernels.GridInterpolationKernel(
                gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(ard_num_dims=2)),
                num_dims=2, grid_size=100
            )
            self.feature_extractor = feature_extractor

            # This module will scale the NN features so that they're nice values
            self.scale_to_bounds = gpytorch.utils.grid.ScaleToBounds(-1., 1.)

        def forward(self, x):
            # We're first putting our data through a deep net (feature extractor)
            projected_x = self.feature_extractor(x)
            projected_x = self.scale_to_bounds(projected_x)  # Make the NN values "nice"

            mean_x = self.mean_module(projected_x)
            covar_x = self.covar_module(projected_x)
            return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

likelihood = gpytorch.likelihoods.GaussianLikelihood()
model = GPRegressionModel(train_x, train_y.to(torch.float32), likelihood)

if torch.cuda.is_available():
    model = model.cuda()
    likelihood = likelihood.cuda()

training_iterations = 300

# Find optimal model hyperparameters
model.train()
likelihood.train()

# Use the adam optimizer
optimizer = torch.optim.Adam([
    {'params': model.feature_extractor.parameters()},
    {'params': model.covar_module.parameters()},
    {'params': model.mean_module.parameters()},
    {'params': model.likelihood.parameters()},
], lr=0.01)

# "Loss" for GPs - the marginal log likelihood
mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

def train():
    iterator = tqdm.notebook.tqdm(range(training_iterations))
    for i in iterator:
        # Zero backprop gradients
        optimizer.zero_grad()
        # Get output from model
        output = model(train_x)
        # Calc loss and backprop derivatives
        loss = -mll(output, train_y)
        loss.backward()
        iterator.set_postfix(loss=loss.item())
        optimizer.step()
        print('loss=',loss)

train()

model.eval()
likelihood.eval()
with torch.no_grad(), gpytorch.settings.use_toeplitz(False), gpytorch.settings.fast_pred_var():
    preds = model(test_x.cuda())
plt.scatter(test_y2, preds.mean.cpu().detach().numpy())
