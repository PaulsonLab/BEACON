#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  1 12:50:40 2024

@author: tang.1856
"""
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision.utils import save_image
# import cv2
import matplotlib.pyplot as plt
# from keras.datasets import mnist
from botorch.models import KroneckerMultiTaskGP, HigherOrderGP
from botorch.optim import optimize_acqf
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.fit import fit_gpytorch_mll
from botorch.utils.transforms import t_batch_mode_transform
from botorch.acquisition import AcquisitionFunction, AnalyticAcquisitionFunction
from botorch.models.transforms.outcome import Standardize
from botorch.models.higher_order_gp import FlattenedStandardize
from linear_operator.settings import _fast_solves
from botorch.optim.fit import fit_gpytorch_mll_torch
from torch.optim import Adam
from functools import partial
from gpytorch.kernels import MaternKernel, RFFKernel, ScaleKernel
from botorch.models import SingleTaskGP, SaasFullyBayesianSingleTaskGP, ModelListGP
from gpytorch.mlls.sum_marginal_log_likelihood import SumMarginalLogLikelihood
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
dim_latent = 8  
N_output = 196
# class CustomAcquisitionFunction(AcquisitionFunction):
#     def __init__(self, model, sampled_behavior, k_NN=10):
#         '''Inits acquisition function with model.'''
#         super().__init__(model=model)
        
#         self.model = model
#         self.k_NN = k_NN
#         self.sampled_behavior = sampled_behavior
        
#     # @t_batch_mode_transform(expected_q=1)
#     def forward(self, X):
#         """Compute the acquisition function value at X."""
        
#         posterior = self.model.posterior(X)
#         samples = posterior.rsample(sample_shape=torch.Size([1])).squeeze(2).squeeze(0).to(device) # Thompson Sampling
        
#         # samples = torch.cat((samples_x, samples_y),dim=1)
#         dist = torch.cdist(samples.to(torch.float64).view(len(samples), 196), self.sampled_behavior.to(torch.float64).view(len(self.sampled_behavior), 196))
#         dist, _ = torch.sort(dist, dim = 1) # sort the distance 
#         n = dist.size()[1]
#         E = torch.cat((torch.ones(self.k_NN), torch.zeros(n-self.k_NN)), dim = 0).to(device) # find the k-nearest neighbor
#         dist = dist*E
#         acquisition_values = torch.sum(dist,dim=1)
        
#         return acquisition_values.flatten()

class CustomAcquisitionFunction(AcquisitionFunction):
    def __init__(self, model, sampled_behavior, k_NN=10):
        '''Inits acquisition function with model.'''
        super().__init__(model=model)
        
        self.model = model
        self.k_NN = k_NN
        self.sampled_behavior = sampled_behavior
        
        # self.ts_sampler_list = []
        # for n_output in range(N_output):
        #     self.ts_sampler_list.append(EfficientThompsonSampler(model.models[n_output]))
        #     self.ts_sampler_list[n_output].create_sample()
        
    # @t_batch_mode_transform(expected_q=1)
    def forward(self, X):
        """Compute the acquisition function value at X."""
        
        sample_list = []
        # for n_output in range(N_output):
            # sample_list.append(self.ts_sampler_list[n_output].query_sample(X))
        posterior = self.model.posterior(X)
        samples = posterior.rsample(sample_shape=torch.Size([1])).squeeze(1).squeeze(0)
        
        # samples = torch.cat(sample_list,1)
        
        dist = torch.cdist(samples.to(torch.float64), self.sampled_behavior.to(torch.float64))
        dist, _ = torch.sort(dist, dim = 1) # sort the distance 
        n = dist.size()[1]
        E = torch.cat((torch.ones(self.k_NN), torch.zeros(n-self.k_NN)), dim = 0) # find the k-nearest neighbor
        dist = dist*E
        acquisition_values = torch.sum(dist,dim=1)
        
        return acquisition_values.flatten()
    
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
    
dim_latent = 8 
transform = transforms.Compose([
    transforms.Resize((14, 14)),  # Resize the image to 14x14 pixels
    transforms.ToTensor()         # Convert the image to a tensor
])

trainset = torchvision.datasets.MNIST(root='data', train = True, download = True, transform = transform)
testset = torchvision.datasets.MNIST(root='./data', train = False, download = True, transform = transform)   
trainset_resize, testset_resize = [],[]
 
for i in range(len(testset)):
    testset_resize.append(testset[i][0])
    
testset_resize = torch.stack(testset_resize).squeeze(1) 
    
VAE_trained = VAE()
VAE_trained.load_state_dict(torch.load('/home/tang.1856/BEACON/VAE1.pth',map_location=torch.device('cpu')))
VAE_trained.eval()

cnn_trained = CNN()
cnn_trained.load_state_dict(torch.load('/home/tang.1856/BEACON/CNN2.pth',map_location=torch.device('cpu')))
cnn_trained.eval()

torch.manual_seed(0)
replicate = 20
N_init = 10
N_original = 2000
# mu, std = VAE_trained.encoder(torch.tensor(testset_resize.reshape(-1, 196)[0:10000]))
# X_original = VAE_trained.sampling(mu, std).detach()

# lb = X_original.min(0).values
# ub = X_original.max(0).values
lb = -2
ub = 2
# X_original = (X_original - X_original.min(0).values)/(X_original.max(0).values - X_original.min(0).values)
# X_original = X_original.to(device)
X_original = torch.rand(N_original,dim_latent).to(device)
Y_original = VAE_trained.decoder(lb+(ub-lb)*X_original.cpu()).detach().view(N_original,14,14).to(device)
# Y_original[Y_original>0.5] = 1
# Y_original[Y_original<0.5] = 0
reachability_list = [[] for _ in range(replicate)]

for seed in range(replicate):
    np.random.seed(seed)
    # Fit Multi-task GP
    ids_acquired = np.random.choice(np.arange((len(Y_original))), size=N_init, replace=False)
    mtgp_train_x= X_original[ids_acquired]
    mtgp_train_y = Y_original[ids_acquired]
    
    
    test_output, last_layer = cnn_trained(mtgp_train_y.cpu().view(len(mtgp_train_y),14,14).unsqueeze(1))
    # pred_y = torch.max(test_output, 1)[1].data.numpy().squeeze()
    pred_y = torch.max(test_output, 1).indices[torch.max(test_output, 1).values>=0.99]
    reachability_list[seed].append(len(np.unique(pred_y)))
    
    for iter_BO in range(30):
        torch.cuda.empty_cache()
        
        # mtgp = HigherOrderGP(
        #     mtgp_train_x,
        #     mtgp_train_y,
        #     outcome_transform=FlattenedStandardize(mtgp_train_y.shape[1:]),
        #     # input_transform=Normalize(train_X["ei_hogp_cf"].shape[-1]),
        #     # covar_modules=covar_module,
        #     latent_init="gp",
        # )
        
        # mtgp_mll = ExactMarginalLogLikelihood(mtgp.likelihood, mtgp)
        # # fit_gpytorch_mll(mll=mtgp_mll, optimizer_kwargs={"options": {"maxiter": 30}})
        
        # with _fast_solves(True):
        #     fit_gpytorch_mll_torch(
        #         mtgp_mll, step_limit=2000, optimizer=partial(Adam, lr=0.05)
        #     )
        
        model_list = []
        for nx in range(196):
            # covar_module = ScaleKernel(RBFKernel(ard_num_dims=dim))
            model_list.append(SingleTaskGP(mtgp_train_x.to(torch.float64), mtgp_train_y.view(len(mtgp_train_x),196)[:,nx].unsqueeze(1).to(torch.float64), outcome_transform=Standardize(m=1)))
        mtgp = ModelListGP(*model_list)
        mll = SumMarginalLogLikelihood(mtgp.likelihood, mtgp)
        fit_gpytorch_mll(mll)
        
        custom_acq_function = CustomAcquisitionFunction(mtgp, mtgp_train_y.view(len(mtgp_train_x),196), k_NN=10)
        acquisition_values = custom_acq_function(X_original)
        ids_sorted_by_aquisition = acquisition_values.argsort(descending=True)
        for id_max_aquisition_all in ids_sorted_by_aquisition:
            if not id_max_aquisition_all.item() in ids_acquired:
                id_max_aquisition = id_max_aquisition_all.item()
                break
        
        # id_max_aquisition = np.random.choice(np.arange((len(Y_original))), size=1, replace=False)[0]
        ids_acquired = np.concatenate((ids_acquired, [id_max_aquisition]))
        mtgp_train_x = X_original[ids_acquired]
        mtgp_train_y = Y_original[ids_acquired] 
        
        # test_output, last_layer = cnn_trained(mtgp_train_y.cpu().view(len(mtgp_train_y),14,14).unsqueeze(1))
        test_output, last_layer = cnn_trained(mtgp_train_y.cpu().unsqueeze(1))
        # pred_y = torch.max(test_output, 1)[1].data.numpy().squeeze()
        pred_y = torch.max(test_output, 1).indices[torch.max(test_output, 1).values>=0.99]
        reachability_list[seed].append(len(np.unique(pred_y)))

# torch.save(reachability_list, 'MNIST_reachability_BEACON_50.pt')
