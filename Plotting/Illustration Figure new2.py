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
from scipy.io import loadmat
from scipy.io import savemat



save_path = "/home/tang.1856/BEACON/BEACON/Plotting/illustrative.mat"
loaded_data = loadmat(save_path)

text_size = 16
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
   
    
    fig, axes = plt.subplots(2, 3, figsize=(14, 14))  

    for i, ax in enumerate(axes.flat):  

        if i>2:
            test_x = loaded_data['test_x'][0]
            test_y = loaded_data['test_y'][0]
            train_x = loaded_data[f'train_x_{i-3}'][0]
            train_y = loaded_data[f'train_y_{i-3}'][0]
            true_y = loaded_data[f'true_y_{i-3}'][0]
            candidate = loaded_data[f'candidate_{i-3}'][0]
            pred = loaded_data[f'pred_{i-3}'][0]
            lcb = loaded_data[f'lcb_{i-3}'][0]
            ucb = loaded_data[f'ucb_{i-3}'][0]
            TS_func = loaded_data[f'TS_{i-3}'][0]
        else:
            test_x = loaded_data['test_x'][0]
            test_y = loaded_data['test_y'][0]
            train_x = loaded_data[f'train_x_{i}'][0]
            train_y = loaded_data[f'train_y_{i}'][0]
            true_y = loaded_data[f'true_y_{i}'][0]
            candidate = loaded_data[f'candidate_{i}'][0]
            pred = loaded_data[f'pred_{i}'][0]
            lcb = loaded_data[f'lcb_{i}'][0]
            ucb = loaded_data[f'ucb_{i}'][0]
            TS_func = loaded_data[f'TS_{i}'][0]
        
        if i<3:
            # top row fig
            ax.plot(test_x, test_y, color='black', linewidth=2)
            ax.plot(test_x, pred, label='Posterior mean', linestyle='dotted', linewidth=2)
            ax.fill_between(test_x, lcb, ucb, color='blue',alpha=0.2)
            
            if i==(BO_iter-1):
                ax.legend(loc=2)
    
            ax.plot(test_x, TS_func, label='Thompson sample', color='green',linestyle='dashed', linewidth=2)
            
            ax.scatter(train_x, train_y, s=100,color='blue')
            ax.scatter(candidate, true_y, marker='*',s=200,color='red') 
    
            ax.set_xlim(lb,ub)
            if i==(BO_iter-1):
                ax.legend(loc=2, fontsize=text_size)
                ax.set_title('Iteration 9', fontsize=text_size)
            
            if i==0:
                ax.set_ylabel('f(x)', fontsize=text_size)
                ax.set_title('Iteration 1', fontsize=text_size)
                
            if i==1:
                ax.set_title('Iteration 5', fontsize=text_size)
      
       
        
        # bottom row fig
        
        if i>2:
            ax.plot(test_x, test_y, color='black', label='True function', linewidth=2)
            
            for p in range(11):
                ax.axhline(y=0.8*p, xmin = lb, xmax=ub, color='grey')
                for element in train_y:
                    if element>0.8*p and element <0.8*(p+1):
                        # ax = plt.gca()
                        xlimes = ax.get_xlim()
                        ax.fill_between(xlimes, 0.8*p, 0.8*(p+1), color='lightblue')
                if true_y>0.8*p and true_y <0.8*(p+1):
                    # ax = plt.gca()
                    xlimes = ax.get_xlim()
                    ax.fill_between(xlimes, 0.8*p, 0.8*(p+1), color='lightpink')
                    
            ax.scatter(train_x, train_y, label='Sampling points',s=100, color='blue')
            ax.scatter(candidate, true_y, marker='*',s=200,color='crimson',label='Query point')
            
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
            
            ax.set_xlim(lb,ub)
            
            if i==5:
                ax.legend(loc=2, fontsize=text_size)   
                ax.set_title('Iteration 9', fontsize=text_size)
           
            ax.set_xlabel('x', fontsize=text_size)
            if i==3:
                ax.set_ylabel('f(x)', fontsize=text_size)
                ax.set_title('Iteration 1', fontsize=text_size)
            
            if i==4:
                ax.set_title('Iteration 5', fontsize=text_size)
          
   
   
      
    
    
