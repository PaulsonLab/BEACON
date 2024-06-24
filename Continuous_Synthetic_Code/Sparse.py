#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 19 23:40:43 2024

@author: tang.1856
"""
import torch
import trieste
import numpy as np
import tensorflow as tf
from trieste.objectives import Hartmann6, GramacyLee, Ackley12, Rosenbrock12
from trieste.types import TensorType
from tqdm import tqdm
from typing import Any, Callable, Optional, Type
import tensorflow as tf
import matplotlib.pyplot as plt
from trieste.models.gpflow import (
    SparseVariational,
    build_svgp,
    KMeansInducingPointSelector,
    GaussianProcessRegression,
    build_gpr
)
from trieste.models.optimizer import BatchOptimizer
import time

def reachability_uniformity(behavior, n_bins = 25, obj_lb = -5, obj_ub = 5):
    # behavior = behavior.squeeze(1).numpy()
    num = len(behavior)
    cum_hist, _ = np.histogram(behavior, np.linspace(obj_lb, obj_ub, n_bins + 1))
    cum_hist = cum_hist[np.nonzero(cum_hist)] / (num) # discrete distribution
    cum_hist_uni = np.mean(cum_hist) * np.ones_like(cum_hist) # theoretical uniform distribution

    cum_coverage = len(cum_hist) / n_bins
    
    
    return cum_coverage

# hartmann_6 = Hartmann6.objective
# search_space = Hartmann6.search_space



# def noisy_hartmann_6(
#     x: TensorType,
# ) -> TensorType:  # contaminate observations with Gaussian noise
#     return hartmann_6(x) + tf.random.normal([len(x), 1], 0, 1, tf.float64)





# Xinit = tf.random.uniform([1, 6])

# Xinit = search_space.sample(1)
# Xvars = tf.Variable(Xinit, constraint=lambda x: tf.clip_by_value(x, 0.5, 2.5))

# @tf.function
def closure(TS_func, Xvars, sampled_behavior = None, k_NN=10):
    """
    Passing sample_axis=0 indicates that the 0-th axis of Xnew 
    should be evaluated 1-to-1 with the individuals paths.
    """
    
    samples = TS_func(Xvars) # Thompson sampling
    dist = tf.norm(samples-sampled_behavior, axis=1)
    dist = tf.sort(dist)
    dist = tf.cast(dist, dtype=tf.float32)
    n = dist.shape
    n = int(n[0])
    E = tf.concat([tf.ones(k_NN), tf.zeros(n - k_NN)], axis = 0)
    dist = tf.multiply(dist, E)
    acq = tf.reduce_sum(dist)
    # acq = TS_func(Xvars)
    return -acq

def optimization(TS_func, training_data, k_NN):
    
    tf.random.set_seed(0)
    
    TS_func = trajectory_sampler.get_trajectory()
    val = []
    for element in query_points:
        val.append(float(TS_func(tf.expand_dims(element,axis=0)).numpy()))
    plt.scatter(query_points, val, label='TS')
    
    Xinit = search_space.sample(1)               
    Xvars = tf.Variable(Xinit, constraint=lambda x: tf.clip_by_value(x, 0, 1))
    # Xvars = tf.Variable(Xinit)
    
    optimizer = tf.keras.optimizers.Adam(0.1)
    for step in tqdm(range(100)):
        with tf.GradientTape(watch_accessed_variables=False) as tape:
            tape.watch(Xvars)
            fvals = closure(TS_func, Xvars, training_data.observations, k_NN)

        grads = tape.gradient(fvals, Xvars)
        optimizer.apply_gradients([(grads, Xvars)])
    # Xvars = tf.clip_by_value(Xvars, 0, 1)
    return Xvars, -closure(TS_func, Xvars, training_data.observations, k_NN)


# class BayesianOptimization:
#     def __init__(self, TS_func, search_space, training_data, k_NN):
#         self.TS_func = TS_func
#         self.search_space = search_space
#         self.training_data = training_data
#         self.k_NN = k_NN

#     # @tf.function
#     def closure(self, Xvars):
#         samples = self.TS_func(Xvars)  # Thompson sampling
#         dist = tf.norm(samples - self.training_data.observations, axis=1)
#         dist = tf.sort(dist)
#         dist = tf.cast(dist, dtype=tf.float32)
#         n = dist.shape[0]
#         E = tf.concat([tf.ones(self.k_NN), tf.zeros(n - self.k_NN)], axis=0)
#         dist = tf.multiply(dist, E)
#         acq = tf.reduce_sum(dist)
#         return -acq

#     def optimize(self, num_steps=100):
#         Xinit = self.search_space.sample(1)
#         Xvars = tf.Variable(Xinit, constraint=lambda x: tf.clip_by_value(x, 0, 1))

#         optimizer = tf.keras.optimizers.Adam(0.1)
#         for step in tqdm(range(num_steps)):
#             with tf.GradientTape(watch_accessed_variables=False) as tape:
#                 tape.watch(Xvars)
#                 fvals = self.closure(Xvars)

#             grads = tape.gradient(fvals, Xvars)
#             if grads is None:
#                 print("no gradient")
#                 break
#             optimizer.apply_gradients([(grads, Xvars)])

#         return Xvars, -self.closure(Xvars)

if __name__ == '__main__':
    
    lb = -5 # lower bound for feature
    ub = 5 # upper bound for feature
    dim = 20 # feature dimension
    N_init = 300 # number of initial training data
    replicate = 5 # number of replicates for experiment
    BO_iter = 1000 # number of evaluations
    n_bins = 100 # grid number for calculating reachability
    # TS = 1 # number of TS (posterior sample)
    k_NN = 10 # k-nearest neighbor
    multi_start = 1
    # Specify the minimum/maximum value for each synthetic function as to calculate reachability
         
    obj_lb = 0 # minimum obj value for Ackley
    # obj_ub = 14.3027 # maximum obj value for Ackley
    # obj_ub = 990396.990397
    obj_ub = 1710685.71069
   
    cost_tensor = []
    coverage_tensor = [] # list containing reachability for every itertation
    time_tensor = [] # list containing CPU time requirement per iteration
    
    # function = GramacyLee.objective
    # search_space = GramacyLee.search_space
    
    # function = Ackley12.objective
    # search_space = Ackley12.search_space
    
    function = Rosenbrock12.objective
    search_space = Rosenbrock12.search_space
        
    for seed in range(replicate): 
        start_time = time.time()
        print('seed:',seed)
        np.random.seed(seed)
        tf.random.set_seed(seed)
        initial_query_points = tf.convert_to_tensor(np.random.rand(N_init, dim)) # generate initial training data for GP
        
        # initial_query_points = search_space.sample(N_init)
        observer = trieste.objectives.utils.mk_observer(function)
        training_data = observer(initial_query_points)
        
        # Standardize observation
        mean_obs = tf.reduce_mean(training_data.observations)
        std_obs = tf.math.reduce_std(training_data.observations)
        standardized_obs = (training_data.observations - mean_obs)/std_obs
        training_data_scale = trieste.data.Dataset(training_data.query_points, standardized_obs)
        
        coverage = reachability_uniformity(torch.from_numpy(training_data.observations.numpy()).flatten(), n_bins, obj_lb, obj_ub) # Calculate the initial reachability and uniformity    
        coverage_list = [coverage]    
        cost_list = [0] # number of sampled data excluding initial data
             
        # Start BO loop
        for i in range(BO_iter):        
                        
            # Define variational GP
            gpflow_model = build_svgp(
                training_data_scale, search_space, num_inducing_points=100
            )

            inducing_point_selector = KMeansInducingPointSelector()

            model = SparseVariational(
                gpflow_model,
                num_rff_features=1_000,
                inducing_point_selector=inducing_point_selector,
                optimizer=BatchOptimizer(
                    tf.optimizers.Adam(0.1), max_iter=100, batch_size=50, compile=True
                ),
            )
            
            # Define nominal GP
            # gpflow = build_gpr(training_data_scale, search_space)
            # model = GaussianProcessRegression(gpflow, optimizer=BatchOptimizer(
            #     tf.optimizers.Adam(0.1), max_iter=100, batch_size=50, compile=True
            # ))
            
           
            model.optimize(dataset=training_data_scale)
            trajectory_sampler = model.trajectory_sampler()
            # TS_func = trajectory_sampler.get_trajectory()
            
         
            # x_test = search_space.sample(100)
            # mu = model.predict_y(x_test)
            # mu = mu[0].numpy()
            # plt.scatter(x_test.numpy(), mu, label='mean')
            # plt.scatter(training_data.query_points.numpy(), training_data.observations.numpy(), label='training data')

            # query_points = search_space.sample(500)
            # val = []
            # for element in query_points:
            #     val.append(float(TS_func(tf.expand_dims(element,axis=0)).numpy()))
                
            # plt.scatter(query_points, val, label='TS')
            acq_val = 0
            
            # plt.figure()
            
            for _ in range(multi_start):
                
                TS_func = trajectory_sampler.get_trajectory()
                
                
                # val = []
                # for element in query_points:
                #     val.append(float(TS_func(tf.expand_dims(element,axis=0)).numpy()))
                # plt.scatter(query_points, val, label='TS')
                
                Xinit = search_space.sample(1)               
                Xvars = tf.Variable(Xinit, constraint=lambda x: tf.clip_by_value(x, 0, 1))
                               
                optimizer = tf.keras.optimizers.Adam(0.1)
                for step in tqdm(range(30), disable=True):
                    with tf.GradientTape(watch_accessed_variables=False) as tape:
                        tape.watch(Xvars)
                        fvals = closure(TS_func, Xvars, training_data_scale.observations, k_NN)
    
                    grads = tape.gradient(fvals, Xvars)
                    optimizer.apply_gradients([(grads, Xvars)])
                fvals = -closure(TS_func, Xvars, training_data_scale.observations, k_NN)
                
                # try:
                #     Xvars, fvals = optimization(TS_func, training_data, k_NN)
                # except:
                #     print('Fail to optimize')
                # # optimizer1 = optimizer(TS_func, training_data, k_NN)
                # # Xvars, fvals = optimizer1.optimization()
                
              
                if fvals>acq_val:
                    acq_val = fvals             
                    candidate = Xvars
            
            # plt.scatter(candidate.numpy(), TS_func(candidate).numpy(), label='minimum of TS')
            # plt.legend()
            
            updated_query_points = tf.concat([training_data_scale.query_points, candidate], axis = 0)
            training_data = observer(updated_query_points)
           
            # Standardize observation
            mean_obs = tf.reduce_mean(training_data.observations)
            std_obs = tf.math.reduce_std(training_data.observations)
            standardized_obs = (training_data.observations - mean_obs)/std_obs
            training_data_scale = trieste.data.Dataset(training_data.query_points, standardized_obs)
            
            coverage = reachability_uniformity(torch.from_numpy(training_data.observations.numpy()).flatten(), n_bins, obj_lb, obj_ub)
            coverage_list.append(coverage)      
            cost_list.append(cost_list[-1]+1)
  
        end_time = time.time()
        cost_tensor.append(cost_list)
        coverage_tensor.append(coverage_list)
        time_tensor.append((end_time-start_time)/BO_iter)
        
                
    time_tensor = torch.tensor(time_tensor, dtype=torch.float32) 
    cost_tensor = torch.tensor(cost_tensor, dtype=torch.float32) 
    coverage_tensor = torch.tensor(coverage_tensor, dtype=torch.float32)   
    torch.save(coverage_tensor, '20DRosen_coverage_list_NS_GPflow_sparse.pt')
    torch.save(cost_tensor, '20DRosen_cost_list_NS_GPflow_sparse.pt')  
    torch.save(time_tensor, '20DRosen_time_list_NS_GPflow_sparse.pt')    
    
