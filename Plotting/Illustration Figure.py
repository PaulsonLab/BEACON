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
import torchsort
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
# import torchsort

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
        # dist= torchsort.soft_sort(dist,regularization_strength=1)
        n = dist.size()[1]
        E = torch.cat((torch.ones(self.k), torch.zeros(n-self.k)), dim = 0) # find the k-nearest neighbor
        dist = dist*E
        acquisition_values = torch.sum(dist,dim=1)
      
        
        return acquisition_values.flatten()

class MaxVariance(AnalyticAcquisitionFunction):
    r"""Single-outcome Max Variance Acq.

    Example:
        >>> model = SingleTaskGP(train_X, train_Y)
        >>> MaxVar = MaxVariance(model)
        >>> maxvar = MaxVar(test_X)
    """

    def __init__(
        self,
        model: Model,
        posterior_transform: Optional[PosteriorTransform] = None,
        maximize: bool = True,
    ) -> None:
        r"""Single-outcome Max Variance.

        Args:
            model: A fitted single-outcome GP model (must be in batch mode if
                candidate sets X will be)
            posterior_transform: A PosteriorTransform. If using a multi-output model,
                a PosteriorTransform that transforms the multi-output posterior into a
                single-output posterior is required.
            maximize: If True, consider the problem a maximization problem.
        """
        super().__init__(model=model, posterior_transform=posterior_transform)
        self.maximize = maximize

    @t_batch_mode_transform(expected_q=1)
    def forward(self, X: Tensor) -> Tensor:
        r"""Evaluate the posterior variance on the candidate set X.

        Args:
            X: A `(b1 x ... bk) x 1 x d`-dim batched tensor of `d`-dim design points.

        Returns:
            A `(b1 x ... bk)`-dim tensor of posterior variance values at the
            given design points `X`.
        """
        mean, sigma = self._mean_and_sigma(X)
        
        return sigma**2

if __name__ == '__main__':
    
    lb = -1
    ub = 2
    dim = 1
    N_init = 3
    replicate = 1
    BO_iter = 3
    n_bins = 25
    TS = 1 # number of TS (posterior sample)
    k = 3
    
    # np.random.seed(0)
        
    # train_x = torch.tensor((np.random.rand(N_init, dim)-0.5-lb)/(ub-lb)) # generate the same intial data as the evolutionay-based NS
    train_x = torch.tensor(np.random.rand(N_init, dim))
    train_x = torch.tensor([[0.9575],[0.3762],[0.8241]],dtype=torch.float64)
    # train_x = torch.tensor([[0.04],[0.12],[0.28],[0.4],[0.6],[0.85],[1.0]],dtype=torch.float64)
    # train_x = torch.tensor([[0.02],[0.15],[0.28],[0.4],[0.6],[1.0]],dtype=torch.float64)
    test_x = torch.tensor(np.linspace(0,1,300)).unsqueeze(1)
    
    # train_x = torch.rand(20,2).to(torch.float64)
    # sobol = SobolEngine(dimension=dim, scramble=True, seed=50)
    # train_x = sobol.draw(n=N_init).to(torch.float64)
        
    # function = toy_function(dim=1)
    function = Ackley(dim=1)
    train_y = function(lb+(ub-lb)*train_x).unsqueeze(1)
    test_y = function(lb+(ub-lb)*test_x)
    
    
    for i in range(BO_iter):    
        # plt.figure()
        # plt.hist(train_y.flatten().detach(), bins = 5, range=(0,8))
        
        plt.figure()
        
        covar_module = ScaleKernel(RBFKernel(ard_num_dims=dim)) # select the RBF kernel
        model = SingleTaskGP(train_x, train_y, covar_module=covar_module, outcome_transform=Standardize(m=1))
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        fit_gpytorch_mll(mll)
        model.train_x = train_x
        model.train_y = train_y
        model.covar_module.base_kernel.lengthscale = 0.1
        # model.covar_module.outputscale = 0.9047
        posterior = model.posterior(X=test_x)
        pred = posterior.mean
        var = posterior.variance
        
        lcb = pred-1*torch.sqrt(var)
        ucb = pred+1*torch.sqrt(var)
        plt.plot(lb+(ub-lb)*test_x.detach().numpy(), test_y.detach().numpy(), color='grey', label='True function', linewidth=2)
        plt.plot(lb+(ub-lb)*test_x.detach().numpy(), pred.flatten().detach().numpy(), label='Posterior mean', linestyle='dotted', linewidth=2)
        
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
        plt.scatter(lb+(ub-lb)*train_x.detach().numpy(), train_y.detach().numpy(), label='Sampling points',s=50,color='blue')
        plt.scatter(lb+(ub-lb)*candidate.detach().numpy(), function(lb+(ub-lb)*candidate).detach().numpy(), label='Query point',marker='*',s=100,color='red') 
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
            plt.legend(loc=2)
            
        plt.figure()
        plt.plot(lb+(ub-lb)*test_x.detach().numpy(), acq_list.detach().numpy(),color='crimson', linewidth=2, label=r'$\alpha_\text{BEACON}$(x)')
        plt.scatter((lb+(ub-lb)*test_x[torch.argmax(acq_list)]).detach().numpy(), max(acq_list).detach().numpy(), marker='*',color='red',s=150, label='Query point')
        plt.xlim(lb,ub)
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
        
        if i==(BO_iter-1):
            plt.legend()
            
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
                
        plt.scatter(lb+(ub-lb)*train_x.detach().numpy(), train_y.detach().numpy(), label='Sampling points',s=50, color='blue')
        plt.scatter(lb+(ub-lb)*candidate.detach().numpy(), function(lb+(ub-lb)*candidate).detach().numpy(), marker='*',s=150,color='crimson',label='Query point') 
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
            plt.legend(loc=2)            
            
        
      
        
        # xx = np.linspace(lb,ub,1000)
        # plt.fill_between(xx, 0.0,0.8)
        # plt.xlabel('x')
        # plt.ylabel(r'$\alpha$(x)')
        # custom_acq_function1 = MaxVariance(model)
        # bounds = torch.tensor([[0.0]*dim, [1.0]*dim], dtype=torch.float)  # Define bounds of the parameter space
        # # Optimize the acquisition function (continuous)
        # candidate, acq_value = optimize_acqf(
        #     acq_function=custom_acq_function1,
        #     bounds=bounds,
        #     q=1,  # Number of candidates to generate
        #     num_restarts=10,  # Number of restarts for the optimizer
        #     raw_samples=20,  # Number of initial raw samples to consider
        # )
        
        # plt.scatter(lb+(ub-lb)*candidate.detach().numpy(), function(lb+(ub-lb)*candidate).detach().numpy(),color='purple', label='MaxVar',marker='^',s=100) 
        
        # plt.legend()
        train_x = torch.cat((train_x, candidate))
        y_new = function(lb+(ub-lb)*candidate).unsqueeze(1)
        train_y = torch.cat((train_y, y_new))
      
  
        
      