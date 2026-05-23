#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 15 15:16:27 2024

@author: tang.1856
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 10 13:26:09 2024

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
from botorch.acquisition import AcquisitionFunction
from botorch.optim import optimize_acqf
from scipy.spatial.distance import cdist, jensenshannon
import numpy as np
from torch.quasirandom import SobolEngine
from botorch.test_functions import Rosenbrock, Ackley
import pickle
from botorch.models.transforms.outcome import Standardize
from botorch import fit_gpytorch_mll
import matplotlib.pyplot as plt
import pandas as pd
import math
from gauche.dataloader import MolPropLoader,ReactionLoader
from gpytorch.means import ConstantMean
from gauche.kernels.fingerprint_kernels.tanimoto_kernel import TanimotoKernel
from gpytorch.distributions import MultivariateNormal
from sklearn.model_selection import train_test_split

class TanimotoGP(SingleTaskGP):

    def __init__(self, train_X, train_Y):
        super().__init__(train_X, train_Y, likelihood=GaussianLikelihood(), outcome_transform=Standardize(m=1))
        self.mean_module = ConstantMean()
        self.covar_module = ScaleKernel(base_kernel=TanimotoKernel())
        self.to(train_X)  # make sure we're on the right device/dtype

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)
    
def initialize_model(train_x, train_obj, state_dict=None):
    """
    Initialise model and loss function.

    Args:
        train_x: tensor of inputs
        train_obj: tensor of outputs
        state_dict: current state dict used to speed up fitting

    Returns: mll object, model object
    """

    # define model for objective
    model = TanimotoGP(train_x, train_obj).to(train_x)
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    # load state dict if it is passed
    if state_dict is not None:
        model.load_state_dict(state_dict)

    return mll, model

def reachability_uniformity(behavior, n_bins = 25, obj_lb = -5, obj_ub = 5, n_hist=25):
    behavior = behavior.squeeze(1).numpy()
    num = len(behavior)
    cum_hist, _ = np.histogram(behavior, np.linspace(obj_lb, obj_ub, n_bins + 1))
    cum_hist = cum_hist[np.nonzero(cum_hist)] / (num) # discrete distribution
    cum_hist_uni = np.mean(cum_hist) * np.ones_like(cum_hist) # theoretical uniform distribution

    cum_coverage = len(cum_hist) / n_hist
    cum_uniformity = 1 - jensenshannon(cum_hist, cum_hist_uni, base=2)
    
    return cum_coverage, cum_uniformity
    
        
class CustomAcquisitionFunction_TS():
    def __init__(self, model, sampled_behavior, k=10, TS = 1):
        
        self.model = model
        self.k = k
        self.sampled_behavior = sampled_behavior
        self.TS = TS
    
    def __call__(self, X):
        """Compute the acquisition function value at X.
        Args:
            X: A `N x d`-dim Tensor from which to sample (in the `N`
                dimension)
            num_samples: The number of samples to draw.
            
        """
        # batch_shape x N x m
        X = X.unsqueeze(0)
        posterior = self.model.posterior(X)
        
        # num_samples x batch_shape x N x m
        samples = posterior.rsample(sample_shape=torch.Size([self.TS])).squeeze(1) # Thompson Sampling
        
        
        acquisition_values = []
        for ts in range(self.TS): # different TS sample
            
            dist = torch.cdist(samples[ts], self.sampled_behavior).squeeze(1)
            dist, _ = torch.sort(dist, dim = 1) # sort the distance 
            n = dist.size()[1]
            E = torch.cat((torch.ones(self.k), torch.zeros(n-self.k)), dim = 0) # find the k-nearest neighbor
            dist = dist*E
            acquisition_values.append(torch.sum(dist,dim=1))
            
        acquisition_values = torch.stack(acquisition_values)
        acquisition_values = torch.max(acquisition_values, dim=0).values
        
        return acquisition_values.flatten()
    

if __name__ == '__main__':
    
    dim = 2133
    N_init = 100
    replicate = 20
    n_bins = 25 # number of bins used to calculate the reachability and uniformity 
    k = 10 # number of k-nearest neighbor
    TS = 1 # number of TS samples
    BO_iter = 200
    
    cost_tensor = []
    coverage_tensor = []
    
    # Load the Photoswitch dataset
    loader = MolPropLoader()
    loader.load_benchmark("ESOL")
    # loader.read_csv("/home/tang.1856/Jonathan/Novelty Search/MultiObj/SMILES_Malaria.csv_with_SAScore.csv", "SMILES","y1")
    # We use the fragprints representations (a concatenation of Morgan fingerprints and RDKit fragment features)
    loader.featurize('ecfp_fragprints')
    # loader.featurize('drfp')
    X_original = loader.features
    y_original = loader.labels
           
    y_original = np.reshape(y_original, (np.size(y_original), 1)) 
    X_original = torch.from_numpy(X_original)
    
    # X_original = (X_original - X_original.min(dim=0).values)/(X_original.max(dim=0).values - X_original.min(dim=0).values) # normalize the original input data
    y_original = torch.from_numpy(y_original)
    
    n_hist = float(torch.count_nonzero(torch.histc(y_original,bins=n_bins)))
    obj_lb = y_original.min() # obj minimum
    obj_ub = y_original.max() # obj maximum 
    
    for seed in range(replicate):
        print('seed:',seed)
        np.random.seed(seed)
        ids_acquired = np.random.choice(np.arange((len(y_original))), size=N_init, replace=False)
        all_indices = np.arange(len(y_original))
        remaining_indices = np.setdiff1d(all_indices, ids_acquired) # indices for testing data
        train_x = X_original[ids_acquired]
        train_y = y_original[ids_acquired]
        test_x = X_original[remaining_indices]
        test_y = y_original[remaining_indices]
       
        
        # ids_acquired_test = np.random.choice(np.arange((len(y_original))), size=100, replace=False)
        # test_x = X_original[ids_acquired_test]
        # test_y= y_original[ids_acquired_test]
        
        coverage, uniformity = reachability_uniformity(train_y, n_bins, obj_lb, obj_ub, n_hist) # Calculate the initial reachability and uniformity
        
        coverage_list = [coverage]
        cost_list = [0] # number of sampled data excluding initial data
        
        # Start BO loop
        for i in range(BO_iter):        
            
            mll, model = initialize_model(train_x.to(torch.float64), train_y.to(torch.float64))
            fit_gpytorch_mll(mll)
            
            custom_acq_function = CustomAcquisitionFunction_TS(model, train_y, k=k, TS = TS)
            
            # Optimize the acquisition function (discrete)
            acquisition_values = custom_acq_function(X_original)
            ids_sorted_by_aquisition = acquisition_values.argsort(descending=True)
            for id_max_aquisition_all in ids_sorted_by_aquisition:
                if not id_max_aquisition_all.item() in ids_acquired:
                    id_max_aquisition = id_max_aquisition_all.item()
                    break
                
            ids_acquired = np.concatenate((ids_acquired, [id_max_aquisition]))
            train_x = X_original[ids_acquired]
            train_y = y_original[ids_acquired] 
                        
            coverage, uniformity = reachability_uniformity(train_y, n_bins, obj_lb, obj_ub, n_hist)
            coverage_list.append(coverage)
            cost_list.append(cost_list[-1] + len([id_max_aquisition]))
    
        cost_tensor.append(cost_list)
        coverage_tensor.append(coverage_list)
       
    
    cost_tensor = torch.tensor(cost_tensor, dtype=torch.float32) 
    coverage_tensor = torch.tensor(coverage_tensor, dtype=torch.float32) 
    torch.save(coverage_tensor, 'ESOL_TS_1_coverage_list_NS.pt')
    torch.save(cost_tensor, 'ESOL_TS_1_cost_list_NS.pt')      
    


