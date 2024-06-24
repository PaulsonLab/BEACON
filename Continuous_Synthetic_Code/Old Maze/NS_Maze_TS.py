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
sys.path.append('/home/tang.1856/Jonathan/Hands-on-Neuroevolution-with-Python-master/Chapter6')
sys.path.append('/home/tang.1856/Jonathan/Novelty Search/Continuous_Synthetic_Code')
from maze_NS import Maze
from gpytorch.mlls.sum_marginal_log_likelihood import SumMarginalLogLikelihood
from ThompsonSampling import EfficientThompsonSampler

def reachability_uniformity(behavior, n_bins = 25, obj_lb = -5, obj_ub = 5):
    behavior = behavior.squeeze(1).numpy()
    num = len(behavior)
    cum_hist, _ = np.histogram(behavior, np.linspace(obj_lb, obj_ub, n_bins + 1))
    cum_hist = cum_hist[np.nonzero(cum_hist)] / (num) # discrete distribution
    cum_hist_uni = np.mean(cum_hist) * np.ones_like(cum_hist) # theoretical uniform distribution

    cum_coverage = len(cum_hist) / n_bins
    cum_uniformity = 1 - jensenshannon(cum_hist, cum_hist_uni, base=2)
    
    return cum_coverage, cum_uniformity
    
class CustomAcquisitionFunction(AcquisitionFunction):
    def __init__(self, model, sampled_behavior, k_NN=10, n_seed = 0):
        '''Inits acquisition function with model.'''
        super().__init__(model=model)
        
        self.model = model
        self.k_NN = k_NN
        self.sampled_behavior = sampled_behavior
        self.n_seed = n_seed
        self.ts_sampler1 = EfficientThompsonSampler(model.models[0])
        self.ts_sampler1.create_sample()
        self.ts_sampler2 = EfficientThompsonSampler(model.models[1])
        self.ts_sampler2.create_sample()
        self.sampled_behavior = torch.stack(self.sampled_behavior,dim=2).squeeze(1)
        
    @t_batch_mode_transform(expected_q=1)
    def forward(self, X):
        """Compute the acquisition function value at X."""
        
       
        # # For other data sets we can sample at once
        # # batch_shape x N x m
        # X = X.squeeze(1).unsqueeze(0)
        # torch.manual_seed(self.n_seed) # make sure different starting point optimize the same posterior sample
        # posterior = self.model.posterior(X)
        
        # # num_samples x batch_shape x N x m
        # samples = posterior.rsample(sample_shape=torch.Size([1])).squeeze(1) # Thompson Sampling       
        # # samples = samples.mean(dim=1)
        # dist = torch.cdist(samples[0].to(torch.float64), self.sampled_behavior.to(torch.float64)).squeeze(0)
        # dist, _ = torch.sort(dist, dim = 1) # sort the distance 
        # n = dist.size()[1]
        # E = torch.cat((torch.ones(self.k_NN), torch.zeros(n-self.k_NN)), dim = 0) # find the k-nearest neighbor
        # dist = dist*E
        # acquisition_values = torch.sum(dist,dim=1)
        
        
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
    
   
    dim = 22
    N_init = 30
    replicate = 20
    BO_iter = 300
    n_bins = 25
    TS = 1 # number of TS (posterior sample)
    k_NN = 10
    obj_lb = 0
    obj_ub = 1
  
    
    cost_tensor = []
    coverage_tensor = []
    uniformity_tensor = []
    cumbent_tensor = []
    
    for seed in range(replicate):
        print('seed:',seed)
        # np.random.seed(seed)
        
        # train_x = torch.tensor((np.random.rand(N_init, dim)-0.5-lb)/(ub-lb)) # generate the same intial data as the evolutionay-based NS
        # train_x = torch.tensor(np.random.rand(N_init, dim))
        # test_x = torch.tensor(np.random.rand(10, dim))
        # torch.manual_seed(seed)
        # train_x = torch.rand(N_init,dim).to(torch.float64)
        
        X_opt = -30+60*torch.tensor([[0.0029, 0.2438, 0.3485, 0.5100, 0.6259, 0.1782, 0.1210, 0.1420, 0.9787,
                0.6091, 0.7387, 0.4671, 0.8452, 0.5360, 0.4084, 0.0163, 0.3617, 0.4038,
                0.3541, 0.2491, 0.8930, 0.6666]])
        
        lb = X_opt-7
        ub = X_opt+7
       
        sobol = SobolEngine(dimension=dim, scramble=True,seed=seed)
        train_x =sobol.draw(n=N_init).to(torch.float64)
        # torch.manual_seed(seed)
        # train_x = torch.rand(N_init,dim)
        # train_x = (train_x - train_x.min(dim=0).values)/(train_x.max(dim=0).values-train_x.min(dim=0).values)
        
        # train_y = function((lb+(ub-lb)*train_x)).unsqueeze(1)
        # train_y = []
       
        # for k in range(N_init):
        #     fun = Maze()
        #     train_y.append(fun.run_experiment(lb+(ub-lb)*train_x[k].unsqueeze(0)))
   
        # train_y = torch.tensor(train_y).unsqueeze(1)
        
        distance, train_y1, train_y2 = [], [], []
        for k in range(N_init):
            fun = Maze()
            Dist, x_loc, y_loc = fun.run_experiment(lb+(ub-lb)*train_x[k].unsqueeze(0))
            train_y1.append(x_loc)
            train_y2.append(y_loc)
            distance.append(Dist)
        
        distance = torch.tensor(distance)
        train_y1 = torch.tensor(train_y1).unsqueeze(1)
        train_y2 = torch.tensor(train_y2).unsqueeze(1)
        train_y = [train_y1, train_y2]
    
        best_reward = float(max(distance))
        # coverage, uniformity = reachability_uniformity(train_y, n_bins, obj_lb, obj_ub) # Calculate the initial reachability and uniformity
        
        # coverage_list = [coverage]
        # uniformity_list = [uniformity]
        cost_list = [0] # number of sampled data excluding initial data
        cumbent_list = [best_reward]
        
        # Start BO loop
        for i in range(BO_iter):        
            print(i)
            model_list = []
            for nx in range(2):
                covar_module = ScaleKernel(RBFKernel(ard_num_dims=dim))
                model_list.append(SingleTaskGP(train_x.to(torch.float64), train_y[nx].to(torch.float64), outcome_transform=Standardize(m=1), covar_module=covar_module))
            model = ModelListGP(*model_list)
            # mll = ExactMarginalLogLikelihood(model.likelihood, model)
            mll = SumMarginalLogLikelihood(model.likelihood, model)
            fit_gpytorch_mll(mll)
            model.models[0].train_x = train_x
            model.models[0].train_y = train_y1
            model.models[1].train_x = train_x
            model.models[1].train_y = train_y2
            
            
            # Perform optimization on TS posterior sample
            acq_val_max = 0
            for ts in range(TS):
                custom_acq_function = CustomAcquisitionFunction(model, train_y, k_NN=k_NN, n_seed=torch.randint(low=0,high=int(1e10),size=(1,)))
                
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
            fun = Maze()
            # y_new = fun.run_experiment(lb+(ub-lb)*candidate)        
            # train_y = torch.cat((train_y, y_new))
            Dist_new, x_loc_new, y_loc_new = fun.run_experiment(lb+(ub-lb)*candidate)
            train_y1 = torch.cat((train_y1, torch.tensor(x_loc_new).unsqueeze(0).unsqueeze(0)))
            train_y2 = torch.cat((train_y2, torch.tensor(y_loc_new).unsqueeze(0).unsqueeze(0)))
            train_y = [train_y1, train_y2]
            distance = torch.cat((distance, Dist_new[0]))
            
            # coverage, uniformity = reachability_uniformity(train_y, n_bins, obj_lb, obj_ub)
            # coverage_list.append(coverage)
            # uniformity_list.append(uniformity)
            cost_list.append(cost_list[-1]+1)
            
            if Dist_new>best_reward:
                print('Best Found reward = ', Dist_new)
                best_reward = Dist_new
            
            cumbent_list.append(float(best_reward))
            torch.save(train_x, 'train_x_seed'+str(seed)+'.pt')
            torch.save(distance, 'distance_seed'+str(seed)+'.pt')
            
        cost_tensor.append(cost_list)
        # coverage_tensor.append(coverage_list)
        # uniformity_tensor.append(uniformity_list)
        cumbent_tensor.append(cumbent_list)
       
    
    cost_tensor = torch.tensor(cost_tensor, dtype=torch.float32) 
    # coverage_tensor = torch.tensor(coverage_tensor, dtype=torch.float32) 
    # uniformity_tensor = torch.tensor(uniformity_tensor, dtype=torch.float32)  
    cumbent_tensor = torch.tensor(cumbent_tensor, dtype=torch.float32)  
    # torch.save(coverage_tensor, 'MediumMaze_TS_1_coverage_list_NS.pt')
    # torch.save(uniformity_tensor, 'MediumMaze_TS_1_uniformity_list_NS.pt')
    torch.save(cost_tensor, 'MediumMaze_TS_1_cost_list_NS.pt')  
    torch.save(cumbent_tensor, 'MediumMaze_TS_1_cumbent_list_NS.pt')  
