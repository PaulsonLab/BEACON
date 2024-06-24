#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 11 13:31:13 2024

@author: tang.1856
"""


import torch
import gpytorch
from botorch.models import SingleTaskGP, SaasFullyBayesianSingleTaskGP, ModelListGP
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
from gpytorch.kernels import MaternKernel, RBFKernel, ScaleKernel
import pickle
from botorch.models.transforms.outcome import Standardize
import matplotlib.pyplot as plt
from gpytorch.mlls.sum_marginal_log_likelihood import SumMarginalLogLikelihood
import gymnasium as gym

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
    reward_acc = 0
    for i in range(300):
       
       # action = env.action_space.sample()  # this is where you would insert your policy
       # param = torch.rand(1,dim).flatten()
       action = policy(param, observation['observation'])
       observation, reward, terminated, truncated, info = env.step(action)
       
       if terminated or truncated:
          observation['achieved_goal'] = observation['desired_goal']
         
          break
    # print('reward=',reward)
        
    final_dist = np.linalg.norm(observation['achieved_goal']-observation['desired_goal'])
    Reward = (initial_dist-final_dist)/initial_dist
    env.close()
    return observation['achieved_goal'], Reward

def reachability_uniformity(behavior, n_bins = 25, obj_lb1 = -5, obj_ub1 = 5, obj_lb2 = -5, obj_ub2 = 5):
   
    cell_size1 = (obj_ub1 - obj_lb1) / n_bins
    cell_size2 = (obj_ub2 - obj_lb2) / n_bins
    
    indices_y1 = ((behavior[:, 0] - obj_lb1) / cell_size1).floor().int()
    indices_y2 = ((behavior[:, 1] - obj_lb2) / cell_size2).floor().int()
    grid_indices = torch.stack((indices_y1, indices_y2), dim=1)
    unique_cells = set(map(tuple, grid_indices.tolist()))
    return len(unique_cells)/(n_bins**2)

    
class CustomAcquisitionFunction(AcquisitionFunction):
    def __init__(self, model, sampled_X, k=10):
        '''Inits acquisition function with model.'''
        super().__init__(model=model)
        
        self.model = model
        self.k = k
        self.sampled_X = sampled_X
       
        
    @t_batch_mode_transform(expected_q=1)
    def forward(self, X):
        """Compute the acquisition function value at X."""
        
        dist = torch.norm(X - self.sampled_X, dim=2)
        dist, _ = torch.sort(dist, dim = 1) # sort the distance 
        n = dist.size()[1]

        E = torch.cat((torch.ones(self.k), torch.zeros(n-self.k)), dim = 0) # find the k-nearest neighbor
        dist = dist*E
        acquisition_values = torch.sum(dist,dim=1)

        
        return acquisition_values.flatten()


if __name__ == '__main__':
    
    lb = -1
    ub = 1
    dim = 8
    N_init = 20
    replicate = 20
    BO_iter = 300   
    TS = 1 # number of TS (posterior sample)
    k = 10 # k-nearest neighbor
    n_bins = 10
    obj_lb1 = -5.1
    obj_ub1 = 5.1
    obj_lb2 = -5.1
    obj_ub2 = 5.1
   
    
    cost_tensor = []
    coverage_tensor = []
    uniformity_tensor = []
    cumbent_tensor = []
    
    for seed in range(replicate):
        print('seed:',seed)
        np.random.seed(seed)
        
        # train_x = torch.tensor((np.random.rand(N_init, dim)-0.5-lb)/(ub-lb)) # generate the same intial data as the evolutionay-based NS
        train_X= torch.tensor(np.random.rand(N_init+10, dim))
        # train_x = torch.rand(20,2).to(torch.float64)
        # sobol = SobolEngine(dimension=dim, scramble=True, seed=seed)
        # train_x = sobol.draw(n=N_init).to(torch.float64)
                       
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
    
        # coverage = reachability_uniformity(train_y, n_bins, obj_lb1, obj_ub1, obj_lb2, obj_ub2) # Calculate the initial reachability and uniformity
        
        # coverage_list = [coverage]
        # uniformity_list = [uniformity]
        cost_list = [0] # number of sampled data excluding initial data
        cumbent_list=[float(max(reward_list))]
        
        # Start BO loop
        for i in range(BO_iter):        
            
            model_list = []
            for nx in range(2):
                covar_module = ScaleKernel(RBFKernel(ard_num_dims=dim))
                model_list.append(SingleTaskGP(train_x.to(torch.float64), train_y[:,nx].unsqueeze(1).to(torch.float64), outcome_transform=Standardize(m=1), covar_module=covar_module))
            model = ModelListGP(*model_list)
            mll = SumMarginalLogLikelihood(model.likelihood, model)
            
            
            # Perform optimization on TS posterior sample
            acq_val_max = 0
            for ts in range(TS):
                custom_acq_function = CustomAcquisitionFunction(model, train_x, k=k)
                
                bounds = torch.tensor([[0.0]*dim, [1.0]*dim], dtype=torch.float)  # Define bounds of the parameter space
                # Optimize the acquisition function (continuous)
                candidate_ts, acq_value = optimize_acqf(
                    acq_function=custom_acq_function,
                    bounds=bounds,
                    q=1,  # Number of candidates to generate
                    num_restarts=10,  # Number of restarts for the optimizer
                    raw_samples=20,  # Number of initial raw samples to consider
                )
                
                if acq_value>acq_val_max:
                    acq_val_max = acq_value
                    candidate = candidate_ts
            
            train_x = torch.cat((train_x, candidate))
            Y_next, reward = environment(lb+(ub-lb)*candidate.flatten()) 
            train_y = torch.cat((train_y, torch.tensor(Y_next).unsqueeze(0)))
            reward_list.append(reward)
            
            # coverage = reachability_uniformity(train_y, n_bins, obj_lb1, obj_ub1, obj_lb2, obj_ub2)
            # coverage_list.append(coverage)
            # uniformity_list.append(uniformity)
            cost_list.append(cost_list[-1]+1)
            cumbent_list.append(float(max(reward_list)))
            # if y_new>best_reward:
            #     print('Best Found reward = ', max(train_y))
            #     best_reward = y_new
        
        cost_tensor.append(cost_list)
        cumbent_tensor.append(cumbent_list)
        # uniformity_tensor.append(uniformity_list)
    
        # calculate_rmse(train_x, train_y)
    
    cost_tensor = torch.tensor(cost_tensor, dtype=torch.float32) 
    cumbent_tensor = torch.tensor(cumbent_tensor, dtype=torch.float32) 
    # uniformity_tensor = torch.tensor(uniformity_tensor, dtype=torch.float32)  
    torch.save(cumbent_tensor, 'Maze_cumbent_list_NS_x_space.pt')
    # torch.save(uniformity_tensor, '12DStyTang_uniformity_list_NS_x_space.pt')
    torch.save(cost_tensor, 'Maze_cost_list_NS_x_space.pt')  
