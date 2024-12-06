#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  6 21:39:31 2024

@author: tang.1856
"""

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
# import torchsort
import pickle
from botorch.models.transforms.outcome import Standardize
import matplotlib.pyplot as plt
from gpytorch.kernels import RBFKernel, ScaleKernel
from ThompsonSampling import EfficientThompsonSampler
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
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

class toy_function():
    def __init__(self,dim):
        self.dim = dim
        
    def __call__(self,x):
        A = 1.0
        lambda_ = 5
        B = 0.5*torch.pi*10
        C = 0
        D = 0
        func = A*torch.exp(-lambda_*x[:,0])*torch.sin(B*x[:,0]+C)+D
        
        # func = torch.sin(torch.pi*x[:,0]) + x[:,0]/3
        return func

def reachability_uniformity(behavior, n_bins = 25, obj_lb = -5, obj_ub = 5):
    behavior = behavior.squeeze(1).numpy()
    num = len(behavior)
    cum_hist, _ = np.histogram(behavior, np.linspace(obj_lb, obj_ub, n_bins + 1))
    cum_hist = cum_hist[np.nonzero(cum_hist)] / (num) # discrete distribution
    cum_hist_uni = np.mean(cum_hist) * np.ones_like(cum_hist) # theoretical uniform distribution

    cum_coverage = len(cum_hist) / n_bins
    cum_uniformity = 1 - jensenshannon(cum_hist, cum_hist_uni, base=2)
    
    return cum_coverage, cum_uniformity
    
class CustomAcquisitionFunction(AnalyticAcquisitionFunction):
    def __init__(self, model, sampled_behavior, k=10):
        '''Inits acquisition function with model.'''
        super().__init__(model=model)
        
        self.model = model
        self.k = k
        self.sampled_behavior = sampled_behavior
        
        self.ts_sampler = EfficientThompsonSampler(model)
        
        self.ts_sampler.create_sample()
        
    @t_batch_mode_transform(expected_q=1)
    def forward(self, X):
        """Compute the acquisition function value at X."""
               
        samples = self.ts_sampler.query_sample(X) # Thompson sampling
        dist = torch.cdist(samples, self.sampled_behavior) # Calculate Euclidean distance between TS and all sampled point
        dist, _ = torch.sort(dist, dim = 1) # sort the distance 
       
        n = dist.size()[1]
        E = torch.cat((torch.ones(self.k), torch.zeros(n-self.k)), dim = 0) # find the k-nearest neighbor
        dist = dist*E
        acquisition_values = torch.sum(dist,dim=1)
              
        return acquisition_values.flatten()


if __name__ == '__main__':
    
    lb = -1
    ub = 2
    dim = 1
    N_init = 3
    replicate = 1
    BO_iter = 10
    n_bins = 25
    TS = 1 # number of TS (posterior sample)
    k = 3
   
    train_x = torch.tensor(np.random.rand(N_init, dim))
    train_x = torch.tensor([[0.9575],[0.3762],[0.8241]],dtype=torch.float64)
    test_x = torch.tensor(np.linspace(0,1,300)).unsqueeze(1)
   
    function = Ackley(dim=1)
    train_y = function(lb+(ub-lb)*train_x).unsqueeze(1)
    test_y = function(lb+(ub-lb)*test_x)
    
    # fig, axes = plt.subplots(3, 3, figsize=(20, 20))  # 3 rows, 3 columns
    # Add plots and legends to subplots
    # for i, ax in enumerate(axes.flat):
    
    for i in range(BO_iter):    
       
        plt.figure()
        
        covar_module = ScaleKernel(RBFKernel(ard_num_dims=dim)) # select the RBF kernel
        model = SingleTaskGP(train_x, train_y, covar_module=covar_module, outcome_transform=Standardize(m=1))
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        fit_gpytorch_mll(mll)
        model.train_x = train_x
        model.train_y = train_y
        model.covar_module.base_kernel.lengthscale = 0.1
       
        posterior = model.posterior(X=test_x)
        pred = posterior.mean
        var = posterior.variance
        
        lcb = pred-1*torch.sqrt(var)
        ucb = pred+1*torch.sqrt(var)
        # plt.plot(lb+(ub-lb)*test_x.detach().numpy(), test_y.detach().numpy(), color='grey', label='True function', linewidth=2)
        plt.plot(lb+(ub-lb)*test_x.detach().numpy(), test_y.detach().numpy(), color='black', linewidth=2)
        plt.plot(lb+(ub-lb)*test_x.detach().numpy(), pred.flatten().detach().numpy(), label='Posterior mean', linestyle='dotted', linewidth=2)
        
        # torch.save(lb+(ub-lb)*test_x.detach().flatten(),'test_x.pt')
        # torch.save(test_y.detach().flatten(),'test_y.pt')
        torch.save(pred.flatten(),'pred_0.pt')
        torch.save(lcb.flatten(),'lcb_0.pt')
        torch.save(ucb.flatten(),'ucb_0.pt')
        
        plt.fill_between(lb+(ub-lb)*test_x.flatten().detach().numpy(), lcb.flatten().detach().numpy(), ucb.flatten().detach().numpy(), color='blue',alpha=0.2)
        if i==(BO_iter-1):
            plt.legend(loc=2)
        # Perform optimization on TS posterior sample
        acq_val_max = 0
        for ts in range(TS):
            custom_acq_function = CustomAcquisitionFunction(model, train_y, k=k)
                
            bounds = torch.tensor([[0.0]*dim, [1.0]*dim], dtype=torch.float)  # Define bounds of the parameter space
            # Optimize the acquisition function (continuous)
            candidate, acq_value = optimize_acqf(
                acq_function=custom_acq_function,
                bounds=bounds,
                q=1,  # Number of candidates to generate
                num_restarts=50,  # Number of restarts for the optimizer
                raw_samples=2000,  # Number of initial raw samples to consider
            )
             
        acq_list = custom_acq_function(test_x.unsqueeze(1).to(torch.float32))
        
        TS_func = custom_acq_function.ts_sampler.query_sample(test_x.unsqueeze(1).to(torch.float32))
        plt.plot(lb+(ub-lb)*test_x.detach().numpy(), TS_func.flatten().detach().numpy(), label='Thompson sample', color='green',linestyle='dashed', linewidth=2)
        # plt.scatter(lb+(ub-lb)*train_x.detach().numpy(), train_y.detach().numpy(), label='Sampling points',s=50,color='blue')
        # plt.scatter(lb+(ub-lb)*candidate.detach().numpy(), function(lb+(ub-lb)*candidate).detach().numpy(), label='Query point',marker='*',s=100,color='red') 
        plt.scatter(lb+(ub-lb)*train_x.detach().numpy(), train_y.detach().numpy(), s=100,color='blue')
        plt.scatter(lb+(ub-lb)*candidate.detach().numpy(), function(lb+(ub-lb)*candidate).detach().numpy(),marker='*',s=200,color='red') 
        
        torch.save(lb+(ub-lb)*train_x.detach().flatten(), 'train_x_0.pt')
        torch.save(lb+(ub-lb)*candidate.detach().flatten(), 'candidate_0.pt')
        torch.save(function(lb+(ub-lb)*train_x).unsqueeze(1).detach().flatten(), 'train_y_0.pt')
        torch.save(function(lb+(ub-lb)*candidate).detach().flatten(), 'true_y_0.pt')
        torch.save(TS_func.flatten().detach().numpy(), 'TS_0.pt')
        
        plt.tick_params(axis='both',
                        which='both',
                        width=2)
        
        ax = plt.gca()
        for label in ax.get_xticklabels():
            label.set_fontsize(14)
            # label.set_fontweight('bold')
            
        for label in ax.get_yticklabels():
            label.set_fontsize(14)
            # label.set_fontweight('bold')
            
        # ax.spines['top'].set_linewidth(2)
        # ax.spines['bottom'].set_linewidth(2)
        # ax.spines['left'].set_linewidth(2)
        # ax.spines['right'].set_linewidth(2)
        plt.xlim(lb,ub)
        if i==(BO_iter-1):
            plt.legend(loc=2, fontsize=18)
        
        if i==0:
            plt.ylabel('f(x)', fontsize=14)
        plt.savefig('illustrative_top'+str(i)+'.png')
        
        # plt.figure()
        # plt.plot(lb+(ub-lb)*test_x.detach().numpy(), acq_list.detach().numpy(),color='crimson', linewidth=2, label=r'$\alpha_\text{BEACON}$(x)')
        # plt.scatter((lb+(ub-lb)*test_x[torch.argmax(acq_list)]).detach().numpy(), max(acq_list).detach().numpy(), marker='*',color='red',s=150, label='Query point')
        # plt.xlim(lb,ub)
        # plt.tick_params(axis='both',
        #                 which='both',
        #                 width=2)
        # ax = plt.gca()
        # for label in ax.get_xticklabels():
        #     label.set_fontsize(12)
        #     label.set_fontweight('bold')
            
        # for label in ax.get_yticklabels():
        #     label.set_fontsize(12)
        #     label.set_fontweight('bold')
        # ax.spines['top'].set_linewidth(2)
        # ax.spines['bottom'].set_linewidth(2)
        # ax.spines['left'].set_linewidth(2)
        # ax.spines['right'].set_linewidth(2)
        
        # if i==(BO_iter-1):
        #     plt.legend()
            
        plt.figure()
        plt.plot(lb+(ub-lb)*test_x.detach().numpy(), test_y.detach().numpy(), color='black', label='True function', linewidth=2)
        
        for p in range(11):
            plt.axhline(y=0.8*p, xmin = lb, xmax=ub, color='grey')
            for element in train_y:
                if element>0.8*p and element <0.8*(p+1):
                    ax = plt.gca()
                    xlimes = ax.get_xlim()
                    ax.fill_between(xlimes, 0.8*p, 0.8*(p+1), color='lightblue')
            if function(lb+(ub-lb)*candidate)>0.8*p and function(lb+(ub-lb)*candidate) <0.8*(p+1):
                ax = plt.gca()
                xlimes = ax.get_xlim()
                ax.fill_between(xlimes, 0.8*p, 0.8*(p+1), color='lightpink')
                
        plt.scatter(lb+(ub-lb)*train_x.detach().numpy(), train_y.detach().numpy(), label='Sampling points',s=100, color='blue')
        plt.scatter(lb+(ub-lb)*candidate.detach().numpy(), function(lb+(ub-lb)*candidate).detach().numpy(), marker='*',s=200,color='crimson',label='Query point')
        
        plt.tick_params(axis='both',
                        which='both',
                        width=2)
        ax = plt.gca()
        for label in ax.get_xticklabels():
            label.set_fontsize(12)
            label.set_fontweight('bold')
            
        for label in ax.get_yticklabels():
            label.set_fontsize(12)
            label.set_fontweight('bold')
            
        ax.spines['top'].set_linewidth(2)
        ax.spines['bottom'].set_linewidth(2)
        ax.spines['left'].set_linewidth(2)
        ax.spines['right'].set_linewidth(2)
        plt.xlim(lb,ub)
        if i==(BO_iter-1):
            plt.legend(loc=2, fontsize=18)            
       
        plt.xlabel('x', fontsize=14)
        if i==0:
            plt.ylabel('f(x)', fontsize=14)
        plt.savefig('illustrative_bottom'+str(i)+'.png')
        
        train_x = torch.cat((train_x, candidate))
        y_new = function(lb+(ub-lb)*candidate).unsqueeze(1)
        train_y = torch.cat((train_y, y_new))
      
    
    
    # # List of PNG file paths
    # png_files = ['illustrative_top0.png', 'illustrative_top1.png', 'illustrative_top2.png', 'illustrative_bottom0.png', 'illustrative_bottom1.png', 'illustrative_bottom2.png']
    
    # # Create a figure and set the grid for subplots
    # fig, axes = plt.subplots(2, 3,  gridspec_kw={'wspace': 0, 'hspace': 0})  # 2 rows, 3 columns
    
    # # Iterate through the files and axes to display images
    # for i, ax in enumerate(axes.flat):
    #     img = mpimg.imread(png_files[i])  # Read the image
    #     ax.imshow(img)  # Show the image
    #     ax.axis('off')  # Turn off the axes
    
    # # Adjust spacing between subplots
    # plt.tight_layout(pad=0.1, h_pad=0.1, w_pad=0.1)
    
    # # Save or show the plot
    # plt.show()

from scipy.io import loadmat
from scipy.io import savemat
save_path = "/home/tang.1856/BEACON/BEACON/Plotting/illustrative.mat"
# Create a dictionary to store all data
data_dict = {}

# Iterate through datasets and save data
for i in range(3):
    data_dict['test_x'] = torch.load('test_x.pt').detach().numpy()
    data_dict['test_y'] = torch.load('test_y.pt').detach().numpy()
    data_dict[f'pred_{i}'] = torch.load(f'pred_{i}.pt').detach().numpy()
    data_dict[f'candidate_{i}'] = torch.load(f'candidate_{i}.pt').detach().numpy()
    data_dict[f'lcb_{i}'] = torch.load(f'lcb_{i}.pt').detach().numpy()
    data_dict[f'ucb_{i}'] = torch.load(f'ucb_{i}.pt').detach().numpy()
    data_dict[f'train_x_{i}'] = torch.load(f'train_x_{i}.pt').detach().numpy()
    data_dict[f'train_y_{i}'] = torch.load(f'train_y_{i}.pt').detach().numpy()
    data_dict[f'true_y_{i}'] = torch.load(f'true_y_{i}.pt').detach().numpy()
    data_dict[f'pred_{i}'] = torch.load(f'pred_{i}.pt').detach().numpy()
    data_dict[f'TS_{i}'] = torch.load(f'TS_{i}.pt')
   

# Save the dictionary to a .mat file
savemat(save_path, data_dict)              
      