from tools.analyze import *
from EA_parameters import parameters
import torch
import matplotlib.pyplot as plt

if __name__ == "__main__":
    """Run experiments according to the parameters from parameters.py and analyze the results."""
    # Need to specify other paramters in the parametersss.py file
    dim = 6
    replicate = 1
    
    obj_lb1 = -5.1# minimum obj value for Rosenbrock
    obj_ub1 = 5.1 # maximum objective value for 4D Rosenbrock
    obj_lb2 = -5.1
    obj_ub2 = 5.1

    lb=-5
    ub=5
   
    coverage_list = []
    cost_list = []
    uniformity_list = []
    cumbent_list = []
    for seed in range(replicate):
    
        # run the experiments and save the data
        coverage, cost = run_sequentially(parameters,seed,dim,obj_lb1,obj_ub1,obj_lb2,obj_ub2,lb,ub)
        cost_list.append(cost)
        coverage_list.append(coverage)
        # cumbent_list.append(cumbent)
        # file_path = save_experiment_results(parameters, data)
   
    coverage_list = torch.tensor(coverage_list).to(torch.float32)
    cost_list = torch.tensor(cost_list, dtype=torch.float32)
    
    # torch.save(coverage_list, 'Cluster_coverage_list_DEA.pt')
    # torch.save(cost_list, 'Cluster_cost_list_DEA.pt')
    