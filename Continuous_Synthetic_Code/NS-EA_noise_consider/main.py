from tools.analyze import *
from parameters import parameters
import torch
import matplotlib.pyplot as plt

if __name__ == "__main__":
    """Run experiments according to the parameters from parameters.py and analyze the results."""
    # Need to specify other paramters in the parametersss.py file
    dim = 4
    replicate = 20
    # obj_lb = -5 # minimum objective value
    # obj_ub = 5 # maximum obj value for toy problem
    obj_lb = 0 # minimum obj value for Rosenbrock
    # obj_ub = 270108 # maximum objective value for 4D Rosenbrock
    # obj_ub = 630252.63 # obj maximum for 8D Rosenbrock
    # obj_ub = 990396.990397 # obj maximum for 12D Rosenbrock
    obj_ub = 14.3027 # maximum obj value for Ackley
    # obj_lb = -3.3224 # minimum for 6D Hartmann
    # obj_ub = 0 # maximum for 6D Hartmann
    # obj_lb = -39.16599*dim # minimum obj val for 4D SkyTang
    # obj_ub = 500 # maximum obj val for 4D SkyTang
    # obj_ub = 1000 # maximum obj val for 8D SkyTang
    # obj_ub = 1500
    # obj_ub = 1
    
  
    lb=-5
    ub=5
   
    coverage_list = []
    cost_list = []
    uniformity_list = []
    cumbent_list = []
    for seed in range(replicate):
    
        # run the experiments and save the data
        coverage, uniformity, cost, cumbent = run_sequentially(parameters,seed,dim,obj_lb,obj_ub,lb,ub)
        cost_list.append(cost)
        coverage_list.append(coverage)
        uniformity_list.append(uniformity)
        cumbent_list.append(cumbent)
        # file_path = save_experiment_results(parameters, data)
   
    coverage_list = torch.tensor(coverage_list)
    uniformity_list = torch.tensor(uniformity_list)
    cost_list = torch.tensor(cost_list, dtype=torch.float32)
    cumbent_list = torch.tensor(cumbent_list, dtype=torch.float32)
    
    torch.save(coverage_list, '4DAckley_coverage_list_GA_considernoise_8.pt')
    # torch.save(uniformity_list, 'MediumMaze_uniformity_list_GA_DEA.pt')
    torch.save(cost_list, '4DAckley_cost_list_GA_considernoise_8.pt')
    # torch.save(cumbent_list, 'MediumMaze_cumbent_list_GA_DEA.pt')
    
    coverage_list_mean = torch.mean(coverage_list, dim = 0)
    coverage_std = torch.std(coverage_list, dim=0)
    cumbent_list_mean = torch.mean(cumbent_list, dim = 0)
    cumbent_std = torch.std(cumbent_list, dim=0)
    uniformity_list_mean = torch.mean(uniformity_list, dim = 0)
    uniformity_std = torch.std(uniformity_list, dim=0)
    cost_list_mean = torch.mean(cost_list, dim = 0)
    
    plt.figure()
    plt.plot(cost_list_mean, coverage_list_mean)
    plt.fill_between(cost_list_mean, coverage_list_mean - coverage_std, coverage_list_mean + coverage_std,  alpha=0.5)
    
    plt.figure()
    plt.plot(cost_list_mean, cumbent_list_mean)
    plt.fill_between(cost_list_mean, cumbent_list_mean - cumbent_std, cumbent_list_mean + cumbent_std,  alpha=0.5)
    
    plt.figure()
    plt.plot(cost_list_mean, uniformity_list_mean)
    plt.fill_between(cost_list_mean, uniformity_list_mean - uniformity_std, uniformity_list_mean + uniformity_std,  alpha=0.5)