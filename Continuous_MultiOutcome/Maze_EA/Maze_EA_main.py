from tools.analyze import *
from Maze_EA_parameters import parameters
import torch
import matplotlib.pyplot as plt

if __name__ == "__main__":
    """Run experiments according to the parameters from parameters.py and analyze the results."""
    # Need to specify other paramters in the parametersss.py file
    dim = 8
    replicate = 20
    
    obj_lb1 = -5.1# minimum obj value for Rosenbrock
    obj_ub1 = 5.1 # maximum objective value for 4D Rosenbrock
    obj_lb2 = -5.1
    obj_ub2 = 5.1

    lb=-1
    ub=1
   
    coverage_list = []
    cost_list = []
    uniformity_list = []
    cumbent_list = []
    for seed in range(replicate):
    
        # run the experiments and save the data
        cumbent, cost = run_sequentially(parameters,seed,dim,obj_lb1,obj_ub1,obj_lb2,obj_ub2,lb,ub)
        cost_list.append(cost)
        cumbent_list.append(cumbent)
        
   
    cost_list = torch.tensor(cost_list, dtype=torch.float32)
    cumbent_list = torch.tensor(cumbent_list, dtype=torch.float32)
    
    torch.save(cost_list, 'Maze_cost_list_DEA.pt')
    torch.save(cumbent_list, 'Maze_cumbent_list_DEA.pt')
    
    cumbent_list_mean = torch.mean(cumbent_list, dim = 0)
    cumbent_std = torch.std(cumbent_list, dim=0)
    cost_list_mean = torch.mean(cost_list, dim = 0)
    
    plt.figure()
    plt.plot(cost_list_mean, cumbent_list_mean)
    plt.fill_between(cost_list_mean, cumbent_list_mean - cumbent_std, cumbent_list_mean + cumbent_std,  alpha=0.5)
    
