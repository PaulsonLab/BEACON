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
from botorch.utils.transforms import t_batch_mode_transform
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
import gymnasium as gym
import gymnasium_robotics
gym.register_envs(gymnasium_robotics)

    
def reachability_uniformity(behavior, n_bins = 25, obj_lb1 = -5, obj_ub1 = 5, obj_lb2 = -5, obj_ub2 = 5):
  
    cell_size1 = (obj_ub1 - obj_lb1) / n_bins
    cell_size2 = (obj_ub2 - obj_lb2) / n_bins
    
    indices_y1 = ((behavior[:, 0] - obj_lb1) / cell_size1).floor().int()
    indices_y2 = ((behavior[:, 1] - obj_lb2) / cell_size2).floor().int()
    grid_indices = torch.stack((indices_y1, indices_y2), dim=1)
    unique_cells = set(map(tuple, grid_indices.tolist()))
    return len(unique_cells)/(n_bins**2)

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
    return observation['achieved_goal'], Reward

if __name__ == '__main__':    
    
    dim = 8
    N_init = 20
    BO_iter = 300
    lb = -1
    ub = 1
    n_bins = 10
  
    replicate = 20
    
    cost_list = [[] for _ in range(replicate)]
    coverage_list = [[] for _ in range(replicate)]
    cumbent_list = [[] for _ in range(replicate)]

    
    for seed in range(replicate):
        print('seed:',seed)
        np.random.seed(seed)
        
        train_X = torch.tensor(np.random.rand(N_init+10, dim))
        train_y , train_x, reward_list =[], [],[]
        for element in train_X:
            loc, reward = environment(lb+(ub-lb)*element) 
            if reward<0.9:
                train_x.append(element.tolist())
                train_y.append(loc)
                reward_list.append(reward)
            if len(train_x)>=N_init:
                break
        train_x = torch.tensor(train_x)
        train_y = torch.tensor(train_y)
        cost_list[seed].append(0)
        cumbent_list[seed].append(float(max(reward_list)))
       
        
        for bo_iter in range(BO_iter):
            
            candidate = torch.rand(1,dim)
           
            train_x = torch.cat((train_x, candidate))
            Y_next, reward = environment(lb+(ub-lb)*candidate.flatten()) 
            train_y = torch.cat((train_y, torch.tensor(Y_next).unsqueeze(0)))
            reward_list.append(reward)
            
            cost_list[seed].append(cost_list[seed][-1]+1)
            cumbent_list[seed].append(float(max(reward_list)))
        # torch.save(train_x, 'Maze_MaxVar_train_x_seed'+str(seed)+'.pt')
        # torch.save(train_y, 'Maze_RS_train_y_seed'+str(seed)+'.pt')
        
    cost_tensor = torch.tensor(cost_list, dtype=torch.float32) 
    cumbent_tensor = torch.tensor(cumbent_list, dtype=torch.float32)   
    # torch.save(cumbent_tensor, 'Maze_cumbent_list_MaxVar.pt')
    # torch.save(cost_tensor, 'Maze_cost_list_MaxVar.pt')  
