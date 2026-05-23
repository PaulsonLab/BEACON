from tools.analyze import *
from parameters import parameters
import torch
import matplotlib.pyplot as plt

if __name__ == "__main__":
    """Run experiments according to the parameters from parameters.py and analyze the results."""
    # Need to specify other paramters in the parametersss.py file
    dim = 4
    replicate = 1

    obj_lb = 0 # minimum obj value for Rosenbrock
    obj_ub = 270108 # maximum objective value for 4D Rosenbrock
    # obj_ub = 630252.63 # obj maximum for 8D Rosenbrock
    # obj_ub = 990396.990397 # obj maximum for 12D Rosenbrock
    
    # obj_ub = 14.3027 # maximum obj value for Ackley

    # obj_lb = -39.16599*dim # minimum obj val for 4D SkyTang
    # obj_ub = 500 # maximum obj val for 4D SkyTang
    # obj_ub = 1000 # maximum obj val for 8D SkyTang
    # obj_ub = 1500
   
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
    
   