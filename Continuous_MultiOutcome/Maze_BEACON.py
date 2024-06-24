#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 10 13:26:09 2024

@author: tang.1856
"""
import torch
import gpytorch
from botorch.models import SingleTaskGP, SaasFullyBayesianSingleTaskGP, ModelListGP
from gpytorch.kernels import MaternKernel, RBFKernel, ScaleKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.mlls import ExactMarginalLogLikelihood
from torch.quasirandom import SobolEngine
from botorch.fit import fit_gpytorch_mll,fit_fully_bayesian_model_nuts
from botorch.utils import standardize
import botorch
from typing import Tuple
from botorch.utils.transforms import t_batch_mode_transform
from botorch.acquisition import AcquisitionFunction, AnalyticAcquisitionFunction
from botorch.acquisition.multi_objective.analytic import MultiObjectiveAnalyticAcquisitionFunction
from botorch.optim import optimize_acqf
from scipy.spatial.distance import cdist, jensenshannon
import numpy as np
from torch.quasirandom import SobolEngine
from botorch.test_functions import Rosenbrock, Ackley, Hartmann, StyblinskiTang
import torchsort
import pickle
from botorch.models.transforms.outcome import Standardize
import matplotlib.pyplot as plt
import sys
sys.path.append('/home/tang.1856/Jonathan/Novelty Search')
from maze_NS import Maze
from gpytorch.mlls.sum_marginal_log_likelihood import SumMarginalLogLikelihood
from ThompsonSampling import EfficientThompsonSampler
from sklearn.cluster import KMeans
import gymnasium as gym


def policy(param, state):
    
    p1 = param[0]*state[0] + param[1]*state[1] + param[2]*state[2] + param[3]*state[3]
    p1 = -1+2*torch.sigmoid(p1)
    
    p2 = param[4]*state[0] + param[5]*state[1] + param[6]*state[2] + param[7]*state[3]
    p2 = -1+2*torch.sigmoid(p2)
    
    return [float(p1),float(p2)]
    
def environment(param):    
    
    env = gym.make("PointMaze_Large-v3", continuing_task=False, render_mode = 'rgb_array')    
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

class CustomAcquisitionFunction(AcquisitionFunction):
    def __init__(self, model, sampled_behavior, k_NN=10):
        '''Inits acquisition function with model.'''
        super().__init__(model=model)
        
        self.model = model
        self.k_NN = k_NN
        self.sampled_behavior = sampled_behavior
        
        self.ts_sampler1 = EfficientThompsonSampler(model.models[0])
        self.ts_sampler1.create_sample()
        self.ts_sampler2 = EfficientThompsonSampler(model.models[1])
        self.ts_sampler2.create_sample()
       
        
    @t_batch_mode_transform(expected_q=1)
    def forward(self, X):
        """Compute the acquisition function value at X."""
        
        
        samples_x = self.ts_sampler1.query_sample(X)
        samples_y = self.ts_sampler2.query_sample(X)
        samples = torch.cat((samples_x, samples_y),dim=1)
        dist = torch.cdist(samples.to(torch.float64), self.sampled_behavior.to(torch.float64))
        dist, _ = torch.sort(dist, dim = 1) # sort the distance 
        n = dist.size()[1]
        E = torch.cat((torch.ones(self.k_NN), torch.zeros(n-self.k_NN)), dim = 0) # find the k-nearest neighbor
        dist = dist*E
        acquisition_values = torch.sum(dist,dim=1)
        
        return acquisition_values.flatten()



if __name__ == '__main__':
       
    dim = 8
    N_init = 50
    replicate = 1
    BO_iter = 200
    n_bins = 10
    TS = 1 
    k_NN = 10
    
    lb = -1
    ub = 1
    
    cost_tensor = []
    coverage_tensor = []
    uniformity_tensor = []
    cumbent_tensor = []
    
    for seed in range(replicate):
        reward_list = []
        print('seed:',seed)
        np.random.seed(seed)
        train_X = torch.tensor(np.random.rand(N_init+10, dim))
        train_y1,train_y2 = [],[]
        train_x = []
        
        for element in train_X:
            loc, reward = environment(lb+(ub-lb)*element) 
            if reward<0.9: # need to make sure we do not sample initial point that has reward>0.9
                train_x.append(element.tolist())
                train_y1.append(loc[0])
                train_y2.append(loc[1])
                reward_list.append(reward)
            if len(train_x)>=N_init:
                break
            
        train_x = torch.tensor(train_x)
        train_y1 = torch.tensor(train_y1)
        train_y2 = torch.tensor(train_y2)
        train_y = torch.stack((train_y1, train_y2),dim=1)
            
        best_reward = float(max(reward_list))
        
        cost_list = [0] 
        cumbent_list = [best_reward]
        
        # Start BO loop
        for i in range(BO_iter):        
            
            model_list = []
            for nx in range(2):
                covar_module = ScaleKernel(RBFKernel(ard_num_dims=dim))
                model_list.append(SingleTaskGP(train_x.to(torch.float64), train_y[:,nx].unsqueeze(1).to(torch.float64), outcome_transform=Standardize(m=1), covar_module=covar_module))
            model = ModelListGP(*model_list)
            mll = SumMarginalLogLikelihood(model.likelihood, model)
            try:
                fit_gpytorch_mll(mll)
            except:
                print('Fail to fit GP!')
            model.models[0].train_x = train_x
            model.models[0].train_y = train_y[:,0].unsqueeze(1)
            model.models[1].train_x = train_x
            model.models[1].train_y = train_y[:,1].unsqueeze(1)

            custom_acq_function = CustomAcquisitionFunction(model, train_y, k_NN=k_NN)
                
            bounds = torch.tensor([[0.0]*dim, [1.0]*dim], dtype=torch.float) 
            # Optimize the acquisition function (continuous)
            candidate, acq_value = optimize_acqf(
                acq_function=custom_acq_function,
                bounds=bounds,
                q=1,  
                num_restarts=10,  
                raw_samples=20, 
            )

                           
            train_x = torch.cat((train_x, candidate))
            loc, reward = environment(lb+(ub-lb)*candidate.flatten())            
            train_y = torch.cat((train_y, torch.tensor(loc).unsqueeze(0)))
            reward_list.append(reward)
            
          
            cost_list.append(cost_list[-1]+1)           
            cumbent_list.append(float(max(reward_list)))
            # torch.save(train_x, 'train_x_seed'+str(seed)+'.pt')
            # torch.save(distance, 'distance_seed'+str(seed)+'.pt')
            
        cost_tensor.append(cost_list)
        cumbent_tensor.append(cumbent_list)
           
    cost_tensor = torch.tensor(cost_tensor, dtype=torch.float32)  
    cumbent_tensor = torch.tensor(cumbent_tensor, dtype=torch.float32)  
    torch.save(cost_tensor, 'Maze_TS_1_cost_list_NS.pt')  
    torch.save(cumbent_tensor, 'Maze_TS_1_cumbent_list_NS.pt')  
