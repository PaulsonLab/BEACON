#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug  3 17:09:56 2024

@author: tang.1856
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
from OilSorbent import OilSorbent
from pyDOE2 import lhs


np.random.seed(0)
# ids_acquired_original = np.random.choice(np.arange((len(Y_original))), size=2000, replace=False)
# # X_original = X_original[ids_acquired_original]
# Y_original = Y_original[ids_acquired_original]
# outcomes = Y_original.numpy()

fun = OilSorbent(7)
lb = torch.tensor([3/7, 0.7, 12, 0.12, 0, 16, 0.41])
ub = torch.tensor([7/3, 2, 24, 0.18, 0.2, 28, 1.32])
rand_samp = lhs(7, 5000, random_state=0) # using latin hypercube
X_original = torch.tensor(rand_samp, dtype=torch.double)
Y_original = fun(lb+(ub-lb)*X_original)
Y_original[Y_original<0] = 0
outcomes = Y_original.numpy()
 
# # Plotting
plt.figure(figsize=(3,3),dpi=150)
plt.scatter(outcomes[:, 0], outcomes[:, 1], alpha=0.5, edgecolors='none', s=1)

cut_x = 60000
cut_y = 6000
# Main division lines
# plt.axvline(x=2, color='r', linestyle='--')
# plt.axhline(y=4, color='r', linestyle='--')
# plt.vlines(x=cut_x, ymin=0, ymax=max(outcomes[:, 1]), colors='r')
# plt.hlines(y = cut_y,  xmin=0, xmax=max(outcomes[:, 0]), colors='r')

# upper right
x_start_ur, x_end_ur = cut_x, max(outcomes[:, 0])
y_start_ur, y_end_ur = cut_y, max(outcomes[:, 1])
n_cut_UR = 8
for i in range(1, n_cut_UR):
    plt.vlines(x=x_start_ur + i * (x_end_ur - x_start_ur) / n_cut_UR, ymin=cut_y, ymax= max(outcomes[:, 1]), color='green', linestyle='--')
    plt.hlines(y=y_start_ur + i * (y_end_ur - y_start_ur) / n_cut_UR, xmin=cut_x, xmax= max(outcomes[:, 0]), color='green', linestyle='--')

# Subdivision of the upper left space (9 subspaces)
n_cut_UL = 4
x_start_ul, x_end_ul = min(outcomes[:, 0]), cut_x
y_start_ul, y_end_ul = cut_y, max(outcomes[:, 1])
for i in range(1, n_cut_UL):
    plt.vlines(x=x_start_ul + i * (x_end_ul - x_start_ul) /n_cut_UL, ymin=cut_y, ymax=max(outcomes[:, 1]), color='red', linestyle='--')
    plt.hlines(y=y_start_ul + i * (y_end_ul - y_start_ul) /n_cut_UL, xmin=min(outcomes[:, 0]), xmax=cut_x, color='red', linestyle='--')

# Subdivision of the lower right space (9 subspaces)
n_cut_LR = 4
x_start_lr, x_end_lr = cut_x, max(outcomes[:, 0])
y_start_lr, y_end_lr = min(outcomes[:, 1]), cut_y
for i in range(1, n_cut_LR):
    plt.vlines(x=x_start_lr + i * (x_end_lr - x_start_lr) /n_cut_LR, ymin=min(outcomes[:, 1]), ymax=cut_y, color='orange', linestyle='--')
    plt.hlines(y=y_start_lr + i * (y_end_lr - y_start_lr) /n_cut_LR, xmin=cut_x, xmax=max(outcomes[:, 0]), color='orange', linestyle='--')

# Show plot

plt.xlim(min(outcomes[:, 0]),max(outcomes[:, 0]))
plt.ylim(min(outcomes[:, 1]),max(outcomes[:, 1]))

# # plt.title('Visually Interesting 2D Outcome Space from High-Dimensional Inputs')
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

# plt.xlabel(r'CO$_{\text{2}}$ uptake', fontsize=12, fontweight='bold')
# plt.ylabel(r'CH$_{\text{4}}$ uptake', fontsize=12, fontweight='bold')
# plt.xlabel(r'y$_\textbf{1}$', fontsize=12, fontweight='bold')
# plt.ylabel(r'y$_2$', fontsize=12, fontweight='bold')
# plt.grid(True)


plt.xlabel('Oil adsorption capacity', fontsize='large')
plt.ylabel('Mechanical Strength', fontsize='large')

plt.tight_layout()











