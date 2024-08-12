#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 31 15:07:12 2024

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
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
pixel = 14
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
    


dim_latent = 8
transform = transforms.Compose([
    transforms.Resize((pixel, pixel)),  # Resize the image to 14x14 pixels
    transforms.ToTensor()         # Convert the image to a tensor
])

trainset = torchvision.datasets.MNIST(root='data', train = True, download = True, transform = transform)
testset = torchvision.datasets.MNIST(root='./data', train = False, download = True, transform = transform)
# (xtrain, ytrain), (x_val_pre , y_val) = mnist.load_data()

# trainset = torch.load('/home/tang.1856/Downloads/MNIST_train.pt')
# testset = torch.load('/home/tang.1856/Downloads/MNIST_test.pt')

trainset_resize, testset_resize = [],[]
for i in range(len(trainset)):
    trainset_resize.append(trainset[i][0])
    
for i in range(len(testset)):
    testset_resize.append(testset[i][0])
    
trainset_resize = torch.stack(trainset_resize).squeeze(1)
testset_resize = torch.stack(testset_resize).squeeze(1)

xtrain = trainset_resize.data.numpy()
ytrain = trainset.targets.numpy()
x_val_pre = testset_resize.data[:1000].numpy()
y_val = testset.targets[:1000].numpy()

count = np.zeros(10)
idx = []
for i in range(0, len(ytrain)):
  for j in range(10):
    if(ytrain[i] == j):
      count[j] += 1
      if(count[j]<=1000):
        idx = np.append(idx, i)
        
y_train = ytrain[idx.astype('int')]
x_train_pre = xtrain[idx.astype('int')]
# x_train_pre = xtrain
x_train = x_train_pre
x_val = x_val_pre

# r,_,_ = x_train_pre.shape
# x_train = np.zeros([r,14,14])
# for i in range(r):
#   a = cv2.resize(x_train_pre[i].astype('float32'), (14,14)) # Resizing the image from 28*28 to 14*14
#   x_train[i] = a

# r,_,_ = x_val_pre.shape
# x_val = np.zeros([r,14,14])
# for i in range(r):
#   a = cv2.resize(x_val_pre[i].astype('float32'), (14,14)) # Resizing the image from 28*28 to 14*14
#   x_val[i] = a
  
  
x_train = np.where(x_train > 0.5, 1, 0)
x_val = np.where(x_val > 0.5, 1, 0)
x_train = x_train.astype(np.float32)
x_val = x_val.astype(np.float32)

batch_size = 32
trainloader = torch.utils.data.DataLoader([[x_train[i], y_train[i]] for i in range(len(y_train))], shuffle=True, batch_size=batch_size)
testloader = torch.utils.data.DataLoader([[x_val[i], y_val[i]] for i in range(len(y_val))], shuffle=True, batch_size=100)


class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()

        self.fc1 = nn.Linear(pixel**2, 128) #Encoder
        self.fc21 = nn.Linear(128, dim_latent) #mu
        self.fc22 = nn.Linear(128, dim_latent) #sigma

        self.fc3 = nn.Linear(dim_latent, 128) #Decoder
        self.fc4 = nn.Linear(128, pixel**2)
        
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
        mu, std = self.encoder(x.view(-1, pixel**2))
        z = self.sampling(mu, std)
        return self.decoder(z), mu, std
    
model = VAE()
if torch.cuda.is_available():
    model.cuda()
    
    
optimizer = torch.optim.Adam(model.parameters(), lr = 0.002)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', factor=0.5, 
                             patience=5, threshold=0.001, cooldown=0,
                             min_lr=0.0001, verbose=True)

def loss_function(y, x, mu, std): 
    ERR = F.binary_cross_entropy(y, x.view(-1, pixel**2), reduction='sum')
    KLD = -0.5 * torch.sum(1 + torch.log(std**2) - mu**2 - std**2)
    # KLD = -0.5 * torch.sum(1 + torch.log(std**2) - mu**2 - torch.log(std**2).exp())
    return ERR + KLD, -ERR, -KLD


# count=0
# err_l, kld_l, n_wu, testl, update = [], [], [], [], []
# for epoch in range(1, 60):
    
#     train_loss = 0
#     for batch_idx, (data, _) in enumerate(trainloader):
#         model.train()
#         data = data.cuda()
#         bsize = data.shape[0]
#         recon_batch, mu, std = model(data)
#         loss, err, kld = loss_function(recon_batch, data, mu, std)
#         loss.backward()
#         train_loss += err.item() + kld.item()
#         optimizer.step()
#         optimizer.zero_grad()
#         err_l.append(err.item()/bsize)
#         kld_l.append(kld.item()/bsize)
#         count+=1
#         n_wu.append(count)

#         if (count%100 == 0): # Validating every 100 weight updates
#           model.eval()
#           a, _ = next(iter(testloader))
#           a = a.cuda()
#           trecon, tmu, tstd = model(a)
#           tloss, terr, tkld = loss_function(trecon, a, tmu, tstd)
#           testl.append(terr/100)
#           update.append(count)

#     scheduler.step(train_loss / len(trainloader.dataset))
    
#     print('====> Epoch: {} Average loss: {:.4f}'.format(epoch, train_loss / len(trainloader.dataset)))
#     model.eval()
#     test_loss= 0
#     with torch.no_grad():
#         for data, _ in testloader:
#             data = data.cuda()
#             recon, mu, std = model(data)
#             loss, err, kld = loss_function(recon, data, mu, std)
#             test_loss += err + kld
    
#     test_loss /= len(testloader.dataset)
#     print('====> Test set loss: {:.4f}'.format(test_loss))
     
# torch.save(model.state_dict(), '/home/tang.1856/BEACON/VAE_6D.pth')

model.load_state_dict(torch.load('/home/tang.1856/BEACON/VAE1.pth',map_location=torch.device('cpu')))
# plt.figure(figsize=(5,3), dpi=100)
# plt.plot(n_wu, err_l, 'b', label='Reconstruction error')
# plt.plot(n_wu, kld_l, 'g', label='KL Divergence')
# plt.title('Plotting first and second term of ELBO')
# plt.xlabel('Number of weight updates')
# plt.ylabel('Value')
# plt.legend()

# model.eval()
# for i in range(2):
#   a,t = next(iter(trainloader))
#   a = a.cuda()
#   recon, mu, std = model(a[0])
#   b = recon[0].reshape((14,14))
#   f, axarr = plt.subplots(1,2)
#   axarr[0].imshow(a[0].detach().cpu().numpy())
#   axarr[1].imshow(b.detach().cpu().numpy())
  
for i in range(25,35):  
  x = np.random.normal(0,1, dim_latent)
  x= x.astype(np.float32)
  x=torch.from_numpy(x)
  x= x.cuda()
  recon = model.decoder(x)
  b = recon.reshape((pixel,pixel))
  print(x)
  f, axarr = plt.subplots(1) 
  axarr.imshow(b.detach().cpu().numpy())

model.eval()
for i in range(5): 
  a,t = next(iter(testloader))
  a = a.cuda()
  recon, mu, std = model(a[0])
  b = recon[0].reshape((pixel,pixel))
  f, axarr = plt.subplots(1,2)
  axarr[0].imshow(a[0].detach().cpu().numpy())
  axarr[1].imshow(b.detach().cpu().numpy())




