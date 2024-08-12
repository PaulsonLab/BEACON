#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 12 13:52:57 2024

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
# from botorch import fit_gpytorch_model
import matplotlib.pyplot as plt
import pandas as pd
import math
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F

device = 'cpu'

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(         
            nn.Conv2d(
                in_channels=1,              
                out_channels=16,            
                kernel_size=5,              
                stride=1,                   
                padding=2,                  
            ),                              
            nn.ReLU(),                      
            nn.MaxPool2d(kernel_size=2),    
        )
        self.conv2 = nn.Sequential(         
            nn.Conv2d(16, 32, 5, 1, 2),     
            nn.ReLU(),                      
            nn.MaxPool2d(2),                
        )
        # fully connected layer, output 10 classes
        self.out = nn.Linear(288, 10)
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        # flatten the output of conv2 to (batch_size, 32 * 7 * 7)
        x = x.view(x.size(0), -1)       
        output = self.out(x)
        output = F.softmax(output, dim=1)
        return output, x    # return x for visualization
    
class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()

        self.fc1 = nn.Linear(196, 128) #Encoder
        self.fc21 = nn.Linear(128, dim_latent) #mu
        self.fc22 = nn.Linear(128, dim_latent) #sigma

        self.fc3 = nn.Linear(dim_latent, 128) #Decoder
        self.fc4 = nn.Linear(128, 196)
        
    def encoder(self, x):
        h = torch.tanh(self.fc1(x))
        return self.fc21(h), self.fc22(h) # mu, std
    
    def sampling(self, mu, std): # Reparameterization trick
        eps1 = torch.randn_like(std)
        eps2 = torch.randn_like(std)
        return 0.5*((eps1 * std + mu) + (eps2 * std + mu)) # Using two samples to compute expectation over z

    def decoder(self, z):
        h = torch.tanh(self.fc3(z))
        return torch.sigmoid(self.fc4(h)) 
    
    def forward(self, x):
        mu, std = self.encoder(x.view(-1, 196))
        z = self.sampling(mu, std)
        return self.decoder(z), mu, std

    
        
class CustomAcquisitionFunction():
    def __init__(self, model, sampled_X, k=10):
        
        self.model = model
        self.k = k
        self.sampled_X = sampled_X
        
    
    def __call__(self, X):
        """Compute the acquisition function value at X."""
        
        dist = torch.cdist(X.to(torch.float64), self.sampled_X.to(torch.float64))
        dist, _ = torch.sort(dist, dim = 1) # sort the distance 
        n = dist.size()[1]

        E = torch.cat((torch.ones(self.k), torch.zeros(n-self.k)), dim = 0) # find the k-nearest neighbor
        dist = dist*E
        acquisition_values = torch.sum(dist,dim=1)
        
        return acquisition_values.flatten()
       
if __name__ == '__main__':
    
    dim = 8
    dim_latent =  dim
    N_init = 10
    replicate = 20
    n_bins = 25 
    k =10
    BO_iter = 40
    cnn_prob = 0.99
    
    cost_tensor = []
    coverage_tensor = []
    
    VAE_trained = VAE()
    VAE_trained.load_state_dict(torch.load('/home/tang.1856/BEACON/VAE1.pth',map_location=torch.device('cpu')))
    VAE_trained.eval()

    cnn_trained = CNN()
    cnn_trained.load_state_dict(torch.load('/home/tang.1856/BEACON/CNN2.pth',map_location=torch.device('cpu')))
    cnn_trained.eval()
    
    N_original = 2000
    lb = -1
    ub = 1
    torch.manual_seed(0)
    X_original = torch.rand(N_original,dim_latent).to(device)
    y_original = VAE_trained.decoder(lb+(ub-lb)*X_original.cpu()).detach().view(N_original,14,14).to(device)
      
    # # Case Study 4
    # df = pd.read_csv('/home/tang.1856/Jonathan/Novelty Search/Training Data/rawdata/Nitrogen.csv') # data from Boobier et al.
    # X_original = (df.iloc[:, 1:(1+dim)]).values
    # y_original = df['U_N2 (mol/kg)'].values
    
    # y_original = np.reshape(y_original, (np.size(y_original), 1)) 
    # X_original = torch.from_numpy(X_original)
    
    # X_original = (X_original - X_original.min(dim=0).values)/(X_original.max(dim=0).values - X_original.min(dim=0).values) 
    # y_original = torch.from_numpy(y_original)
    
    # n_hist = float(torch.count_nonzero(torch.histc(y_original,bins=n_bins)))
    
    # obj_lb = y_original.min() # obj minimum
    # obj_ub = y_original.max() # obj maximum 
    
    for seed in range(replicate):
        print('seed:',seed)
        np.random.seed(seed)
        ids_acquired = np.random.choice(np.arange((len(y_original))), size=N_init, replace=False)
        train_x = X_original[ids_acquired]
        train_y = y_original[ids_acquired] # original scale
       
        # coverage, uniformity = reachability_uniformity(train_y, n_bins, obj_lb, obj_ub, n_hist) 
        test_output, last_layer = cnn_trained(train_y.cpu().view(len(train_y),14,14).unsqueeze(1))
        # pred_y = torch.max(test_output, 1)[1].data.numpy().squeeze()
        pred_y = torch.max(test_output, 1).indices[torch.max(test_output, 1).values>=cnn_prob]
        # reachability_list[seed].append(len(np.unique(pred_y)))
        
        coverage_list = [len(np.unique(pred_y))]
        cost_list = [0] 
        
        # Start BO loop
        for i in range(BO_iter):        
            
            # model = SingleTaskGP(train_x, train_y, outcome_transform=Standardize(m=1))
            # mll = ExactMarginalLogLikelihood(model.likelihood, model)
            # fit_gpytorch_mll(mll)
            
            custom_acq_function = CustomAcquisitionFunction(None, train_x, k=k)
            
            # Optimize the acquisition function (discrete)
            acquisition_values = custom_acq_function(X_original)
            ids_sorted_by_aquisition = acquisition_values.argsort(descending=True)
            for id_max_aquisition_all in ids_sorted_by_aquisition:
                if not id_max_aquisition_all.item() in ids_acquired:
                    id_max_aquisition = id_max_aquisition_all.item()
                    break
                
            ids_acquired = np.concatenate((ids_acquired, [id_max_aquisition]))
            train_x = X_original[ids_acquired]
            train_y = y_original[ids_acquired] # original scale
            
            # coverage, uniformity = reachability_uniformity(train_y, n_bins, obj_lb, obj_ub, n_hist)
            test_output, last_layer = cnn_trained(train_y.cpu().view(len(train_y),14,14).unsqueeze(1))
            # pred_y = torch.max(test_output, 1)[1].data.numpy().squeeze()
            pred_y = torch.max(test_output, 1).indices[torch.max(test_output, 1).values>=cnn_prob]
            coverage_list.append(len(np.unique(pred_y)))   
            cost_list.append(cost_list[-1] + len([id_max_aquisition]))
            
        cost_tensor.append(cost_list)
        coverage_tensor.append(coverage_list)
          
    cost_tensor = torch.tensor(cost_tensor, dtype=torch.float32) 
    coverage_tensor = torch.tensor(coverage_tensor, dtype=torch.float32)   
    torch.save(coverage_tensor, 'MNIST_coverage_list_NS_FS.pt')    
    torch.save(cost_tensor, 'MNIST_cost_list_NS_FS.pt')  
   
    


