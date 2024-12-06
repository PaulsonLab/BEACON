#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 31 16:11:45 2024

@author: tang.1856
"""

import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset, TensorDataset
import torch.nn as nn
from torch import optim
from torch.autograd import Variable
import torchvision.transforms as transforms
# import cv2
import numpy as np
import torch.nn.functional as F
import torchvision

class Binarize(object):
    def __call__(self, img):
        return (img > 0.5).float()


# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device
pixel = 14
# train_data = torch.load('/home/tang.1856/Downloads/MNIST_train.pt')
# test_data = torch.load('/home/tang.1856/Downloads/MNIST_test.pt')
transform = transforms.Compose([
    transforms.Resize((pixel, pixel)),  # Resize the image to 14x14 pixels
    transforms.ToTensor(),         # Convert the image to a tensor
    Binarize()
])

train_data = torchvision.datasets.MNIST(root='data', train = True, download = True, transform = transform)
test_data = torchvision.datasets.MNIST(root='./data', train = False, download = True, transform = transform)
# train_data = TensorDataset()
loaders = {
    'train' : torch.utils.data.DataLoader(train_data, 
                                          batch_size=100, 
                                          shuffle=True, 
                                          num_workers=1),
    
    'test'  : torch.utils.data.DataLoader(test_data, 
                                          batch_size=100, 
                                          shuffle=True, 
                                          num_workers=1),
}


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
        # self.out = nn.Softmax(288,10)
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        # flatten the output of conv2 to (batch_size, 32 * 7 * 7)
        x = x.view(x.size(0), -1)       
        output = self.out(x)
        output = F.softmax(output, dim=1)
        return output, x    # return x for visualization
    
cnn = CNN()    
loss_func = nn.CrossEntropyLoss() 

optimizer = optim.Adam(cnn.parameters(), lr = 0.001)   


num_epochs = 20
def train(num_epochs, cnn, loaders):
    
    cnn.train()
        
    # Train the model
    total_step = len(loaders['train'])
        
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(loaders['train']):
            
            # gives batch data, normalize x when iterate train_loader
            b_x = Variable(images)   # batch x
            b_y = Variable(labels)   # batch y
            output = cnn(b_x)[0]               
            loss = loss_func(output, b_y)
            
            # clear gradients for this training step   
            optimizer.zero_grad()           
            
            # backpropagation, compute gradients 
            loss.backward()    
            # apply gradients             
            optimizer.step()                
            
            if (i+1) % 100 == 0:
                print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch + 1, num_epochs, i + 1, total_step, loss.item()))
                pass
        
        pass
    
    
    pass

# train(num_epochs, cnn, loaders)
cnn.load_state_dict(torch.load('/home/tang.1856/BEACON/BEACON/MNIST/CNN2.pth',map_location=torch.device('cpu')))

def test():
    # Test the model
    cnn.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in loaders['test']:
            test_output, last_layer = cnn(images)
            pred_y = torch.max(test_output, 1)[1].data.squeeze()
            accuracy = (pred_y == labels).sum().item() / float(labels.size(0))
            pass
        print('Test Accuracy of the model on the 10000 test images: %.2f' % accuracy)
    
    pass

test()

sample = next(iter(loaders['test']))
imgs, lbls = sample

actual_number = lbls[:10].numpy()


test_output, last_layer = cnn(imgs[:10])
pred_y = torch.max(test_output, 1)[1].data.numpy().squeeze()
print(f'Prediction number: {pred_y}')
print(f'Actual number: {actual_number}')

# torch.save(cnn.state_dict(), '/home/tang.1856/BEACON/CNN2.pth')




















