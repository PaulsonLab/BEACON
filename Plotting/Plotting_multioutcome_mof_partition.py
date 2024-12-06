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

file_path1 = '/home/tang.1856/Downloads/PMOF20K_traindata_7000_train.csv'
data1 = pd.read_csv(file_path1)
y1 = data1['pure_uptake_CO2_298.00_15000']
y2 = data1['pure_uptake_methane_298.00_580000']

Y_original = torch.stack((torch.tensor(y1.values),torch.tensor(y2.values)),dim=1)
np.random.seed(0)
ids_acquired_original = np.random.choice(np.arange((len(Y_original))), size=2000, replace=False)
# X_original = X_original[ids_acquired_original]
Y_original = Y_original[ids_acquired_original]
outcomes = Y_original.numpy()

# # Plotting
plt.figure(dpi=150)
plt.scatter(outcomes[:, 0], outcomes[:, 1], alpha=0.5, edgecolors='none', s=4)

cut_x = 1
cut_y = 4
# Main division lines
# plt.axvline(x=2, color='r', linestyle='--')
# plt.axhline(y=4, color='r', linestyle='--')
plt.vlines(x=cut_x, ymin=0, ymax=max(outcomes[:, 1]), colors='r')
plt.hlines(y = cut_y,  xmin=0, xmax=max(outcomes[:, 0]), colors='r')

x_start_ur, x_end_ur = cut_x, max(outcomes[:, 0])
y_start_ur, y_end_ur = cut_y, max(outcomes[:, 1])
for i in range(1, 4):
    plt.vlines(x=x_start_ur + i * (x_end_ur - x_start_ur) / 4, ymin=cut_y, ymax= max(outcomes[:, 1]), color='gray', linestyle='--')
    plt.hlines(y=y_start_ur + i * (y_end_ur - y_start_ur) / 4, xmin=cut_x, xmax= max(outcomes[:, 0]), color='gray', linestyle='--')

# Subdivision of the upper left space (9 subspaces)
x_start_ul, x_end_ul = min(outcomes[:, 0]), cut_x
y_start_ul, y_end_ul = cut_y, max(outcomes[:, 1])
for i in range(1, 3):
    plt.vlines(x=x_start_ul + i * (x_end_ul - x_start_ul) / 3, ymin=cut_y, ymax=max(outcomes[:, 1]), color='gray', linestyle='--')
    plt.hlines(y=y_start_ul + i * (y_end_ul - y_start_ul) / 3, xmin=min(outcomes[:, 0]), xmax=cut_x, color='gray', linestyle='--')

# Subdivision of the lower right space (9 subspaces)
x_start_lr, x_end_lr = cut_x, max(outcomes[:, 0])
y_start_lr, y_end_lr = min(outcomes[:, 1]), cut_y
for i in range(1, 3):
    plt.vlines(x=x_start_lr + i * (x_end_lr - x_start_lr) / 3, ymin=min(outcomes[:, 1]), ymax=cut_y, color='gray', linestyle='--')
    plt.hlines(y=y_start_lr + i * (y_end_lr - y_start_lr) / 3, xmin=cut_x, xmax=max(outcomes[:, 0]), color='gray', linestyle='--')

# Show plot

plt.xlim(min(outcomes[:, 0]),max(outcomes[:, 0]))
plt.ylim(min(outcomes[:, 1]),max(outcomes[:, 1]))

plt.xlabel('Oil adsorption capacity', fontsize='large')
plt.ylabel('Mechanical strength', fontsize='large')
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
plt.show()
