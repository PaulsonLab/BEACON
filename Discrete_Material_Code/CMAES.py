#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug  4 18:13:06 2024

@author: tang.1856
"""

import torch
import numpy as np
import cma



def reachability_uniformity(behavior, n_bins = 25, obj_lb = -5, obj_ub = 5, n_hist = 25):
    behavior = behavior.squeeze(1).numpy()
    num = len(behavior)
    cum_hist, _ = np.histogram(behavior, np.linspace(obj_lb, obj_ub, n_bins + 1))
    cum_hist = cum_hist[np.nonzero(cum_hist)] / (num) # discrete distribution
    cum_hist_uni = np.mean(cum_hist) * np.ones_like(cum_hist) # theoretical uniform distribution

    cum_coverage = len(cum_hist) / n_hist
    # cum_uniformity = 1 - jensenshannon(cum_hist, cum_hist_uni, base=2)
    
    return cum_coverage

class F_int:
    def __init__(self):
        
        # self.X_original = X_original
        # self.Y_original = Y_original
        self.loop = 0
        
    def __call__(self, x, X_original, Y_original):
        # self.loop+=1
        # if self.loop%50==0:
        #     print(self.loop)
        # x = np.floor(x)
        comparison = np.all(X_original.astype(np.int32) == x.astype(np.int32), axis = 1)
        matching_indices = np.where(comparison)[0][0]
        fun_val = Y_original[matching_indices]
        
        # fun_val = float(self.Y_original[x])
        return float(fun_val)

if __name__ == '__main__':

    replicate = 5
    n_bins = 50
    N_init = 50
    
    
    reachability_list = [[] for _ in range(replicate)]
    cost_list = [[] for _ in range(replicate)]
    
    for seed in range(replicate):
        np.random.seed(0)
        ids_acquired_original = np.random.choice(np.arange((65792)), size=10000, replace=False)
        X_original= np.load('/home/tang.1856/BEACON/tf_bind_8-SIX6_REF_R1/tf_bind_8-x-0.npy')[ids_acquired_original]
        Y_original = -np.load('/home/tang.1856/BEACON/tf_bind_8-SIX6_REF_R1/tf_bind_8-y-0.npy')[ids_acquired_original]
        obj_lb = Y_original.min() # obj minimum
        obj_ub = Y_original.max() # obj maximum 
        n_hist = float(torch.count_nonzero(torch.histc(torch.tensor(Y_original),bins=n_bins)))
        
        cost_list[seed].append(0)
        np.random.seed(seed)
        ids_initial = np.random.choice(np.arange((len(Y_original))), size=N_init, replace=False)
        initial_indice = np.argmin(Y_original[ids_initial])
        
        
        x0 = X_original[initial_indice] # starting point for CMAES
        mask = torch.ones(torch.tensor(X_original).size(0), dtype=torch.bool)
        mask[ids_initial] = False  # Set False for indices to discard
        sampled_Y = torch.tensor(Y_original[ids_initial].flatten())
        X_original = X_original[mask]
        Y_original = Y_original[mask]
        reachability_initial = reachability_uniformity(sampled_Y.unsqueeze(1), n_bins=n_bins, obj_lb=obj_lb, obj_ub=obj_ub, n_hist=n_hist)
        reachability_list[seed].append(reachability_initial)
        
        sigma0 = 0.5
        F_init = F_int()
        # x, es = cma.fmin2(F_int(X_original, Y_original), x0, 1,
        #                   {'integer_variables': list(range(len(x0))),
        #                     'bounds': [0, 4 - 1e-9],
        #                     'tolflatfitness':9})  
        
        es = cma.CMAEvolutionStrategy(x0, sigma0, options={'bounds': [0, 4 - 1e-9], 'tolflatfitness':9})
        
        # ids_acquired = torch.tensor([initial_indice])
        # sampled_Y = Y_original
        while not es.stop():
            solutions = es.ask()
            distances = torch.cdist(torch.tensor(solutions).to(torch.float64), torch.tensor(X_original).to(torch.float64))
            min_indices = torch.argmin(distances, dim=1)
            # ids_acquired = torch.cat((ids_acquired, min_indices))
            solutions = torch.tensor(X_original)[min_indices].numpy()
           
            # reachability_uniformity(Y_original[ids_acquired], n_bins=n_bins, obj_lb=obj_lb, obj_ub=obj_ub)
            es.tell(solutions, [F_init(x, X_original, Y_original) for x in solutions])
            sampled_Y = torch.cat((sampled_Y, torch.tensor([F_init(x, X_original, Y_original) for x in solutions])))
            reachability_list[seed].append(reachability_uniformity(sampled_Y.unsqueeze(1), n_bins=n_bins, obj_lb=obj_lb, obj_ub=obj_ub, n_hist=n_hist))
            
            es.logger.add()
            es.disp()
            
            mask = torch.ones(torch.tensor(X_original).size(0), dtype=torch.bool)
            mask[min_indices] = False  # Set False for indices to discard
            X_original = X_original[mask]
            Y_original = Y_original[mask]
            
            cost_list[seed].append(len(sampled_Y)-N_init)
            if len(sampled_Y)>349:
                break
            
    torch.save(torch.tensor(reachability_list), 'DNA_coverage_list_CMAES.pt')
    torch.save(torch.tensor(cost_list).to(torch.float64), 'DNA_cost_list_CMAES.pt')
            
    
    
