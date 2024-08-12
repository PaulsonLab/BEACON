import os
import time
import multiprocessing as mp
import _pickle as pickle
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist, jensenshannon
from botorch.test_functions import Rosenbrock, Ackley, Hartmann, StyblinskiTang
import torch
import sys
# sys.path.append('/home/tang.1856/Jonathan/Hands-on-Neuroevolution-with-Python-master/Chapter6')
# from maze_NS import Maze
from torch.quasirandom import SobolEngine
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F

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
cnn_trained = CNN()
cnn_trained.load_state_dict(torch.load('/home/tang.1856/BEACON/CNN2.pth',map_location=torch.device('cpu')))
cnn_trained.eval()

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

class Experiment:
    """
    Novelty search in a simple simulated setting where a 2D genome is mapped to a 1D behavior space via a simple  non-linear mapping.
    """

    def __init__(self, params, display=False, seed = None, dim = 2, obj_lb = -5, obj_ub = 5, lb = -5, ub = 5):
        """Constructor

        Args:
            params: dictionary with the following keys
                mapping (str): type of genome-to-behavior mapping {'linear', 'hyperbolic', 'bounded_linear', 'linear_seesaw', 'multiplicative',
                                                                   'soft_multiplicative', 'hyperbolic_seesaw', 'multiplicative_seesaw', 'cosinus',
                                                                   '2D_cosinus', 'multiplicative_cosinus', 'peaks', '2D_peaks'}
                eta (float): mutation spread parameter
                n_pop (int): number of individuals in the population
                n_offspring (int): number of offsprings generated from the population at each generation
                criterion (str): type of novelty computation {'novelty', 'hull', 'fitness', 'random'}
                n_neighbors (int): number of closest neighbors to compute the "novelty" criterion)
                best_fit (float): arbitrary behavior with the maximum fitness for the "fitness" criterion
                n_selected (int): number of offsprings added to the archive at each generation
                n_evolvability (int): number of samples generated from each genome to evaluate its evolvability
                n_epochs (int): number of generations of the search process
                addition (str): strategy to add individuals to the archive {'novelty', 'random'}
                restart (int): generation at which the population is re-initialized
                frozen (int): generation at which the reference set for the novelty computation is frozen
            display (bool): display the state of the search at each generation

        """
        assert params["mapping"] in ['linear', 'hyperbolic', 'bounded_linear', 'linear_seesaw', 'multiplicative',
                                     'soft_multiplicative', 'hyperbolic_seesaw', 'multiplicative_seesaw', 'cosinus',
                                     '2D_cosinus', 'multiplicative_cosinus', 'peaks', '2D_peaks', 'Rosen', 'Ackley', 'Hartmann', 'StyTang','medium maze','MNIST'], "incorrect type of mapping"
        assert params["eta"] > 0, "eta must be greater than 0"
        assert params["n_pop"] > 0, "n_pop must be greater than 0"
        assert params["n_offspring"] > 0, "n_offspring must be greater than 0"
        assert params["criterion"] in ['novelty', 'hull', 'fitness', 'random'], "incorrect selection criterion"
        assert params["n_neighbors"] > 0, "n_neighbors must be greater than 0"
        assert params["n_selected"] > 0, "n_selected must be greater than 0"
        assert params["n_evolvability"] > 0, "n_evolvability must be greater than 0"
        assert params["n_epochs"] > 0, "n_epochs must be greater than 0"
        assert params["addition"] in ['novelty', 'random'], "incorrect addition criterion"
        assert params["restart"] > 0, "restart must be greater than 0"
        assert params["frozen"] > 0, "frozen must be greater than 0"
        
        # indexation
        self.I_PARENT = 0  # index of the parent in the previous generation
        self.I_GENERATION = 1  # generation at which the agent is created
        self.I_SELECTED_POP = 2  # flag indicating if the individuals is selected to be part of the next population
        self.I_SELECTED_ARC = 3  # flag indicating if the individuals is selected to be part of the archive
        self.I_AGE = 4  # age of the individuals = number of generations since its creation
        self.I_GENOME = [5+k for k in range(dim)]  # 2D genome
        self.I_BEHAVIOR = 5+dim  # 1D behavior
        self.I_DISTANCE = 5+dim+1+195  # distance to the parent when the individual is created
        self.I_NOVELTY = 5+dim+2+195  # novelty of the individual
        self.I_COVERAGE = 5+dim+3+195  # coverage associated with the genome
        self.I_UNIFORMITY = 5+dim+4+195  # uniformity associated with the genome
        self.I_CUM_COVERAGE = 5+dim+5+195  # cumulative coverage associated with the current group of individuals
        self.I_CUM_UNIFORMITY = 5+dim+6+195  # cumulative uniformity associated with the current group of individuals
        self.SIZE_INDIVIDUAL = 5+dim+7+195  # length of the descriptor for each individual
        
        # experiment parameters
        self.seed = seed
        self.dim = dim
        self.mapping = params["mapping"]
        self.eta = params["eta"]
        self.n_pop = params["n_pop"]
        self.n_offspring = params["n_offspring"]
        self.criterion = params["criterion"]
        self.n_neighbors = params["n_neighbors"]
        self.best_fit = params["best_fit"]
        self.n_selected = params["n_selected"]
        self.n_evolvability = params["n_evolvability"]
        self.n_epochs = params["n_epochs"]
        self.addition = params["addition"]
        self.restart = np.inf if params["restart"] is None else params["restart"]
        self.frozen = np.inf if params["frozen"] is None else params["frozen"]

        self.display = display

        # preallocate memory - individuals are stored as a vector of float, which is faster in access than storing them as a class instances
        self.archive = np.full((self.n_epochs + 1, self.n_selected, self.SIZE_INDIVIDUAL), np.nan, dtype=np.float32)
        self.population = np.full((self.n_epochs + 1, self.n_pop, self.SIZE_INDIVIDUAL), np.nan, dtype=np.float32)
        self.offsprings = np.full((self.n_epochs, self.n_offspring, self.SIZE_INDIVIDUAL), np.nan, dtype=np.float32)
        
        self.lb = lb
        self.ub = ub
        self.n_bins = params["n_bins"]
        
        self.obj_lb = obj_lb
        self.obj_ub = obj_ub
        # iterator
        self.t = 0

    def gene_to_behavior(self, g):
        """Non-linear mapping from genome to behavior in [-5, 5]

        Args:
            g (np.array((N, d), float)): genomes

        Returns:
            np.array((N), float): behaviors

        """
        assert all(np.abs(g.flatten()) <= 5.), "the gene values should be in [-5., 5.]"

        VAE_trained = VAE()
        VAE_trained.load_state_dict(torch.load('/home/tang.1856/BEACON/VAE1.pth',map_location=torch.device('cpu')))
        VAE_trained.eval()
            
        if self.mapping=='Rosen':
            func = Rosenbrock(dim=self.dim)
            behavior = func(self.lb+(self.ub-self.lb)*torch.tensor(g)).numpy()
            
    
        elif self.mapping == 'medium maze':
            behavior = []
            for w in range(len(g)):
                func = Maze()
                behavior.append(func.run_experiment(self.lb+(self.ub-self.lb)*torch.tensor(g[w])).numpy())
            behavior = np.array(behavior).flatten()
        elif self.mapping == 'MNIST':
            behavior = VAE_trained.decoder(self.lb+(self.ub-self.lb)*torch.tensor(g)).detach().numpy()
        else:
            behavior = 0 * g[:, 0]
        return behavior

    def mutate_genome(self, g, low=0.0, high=1.0):
        """Mutation operator

        Args:
            g (np.array((N, d), float)): genomes
            low (float): lower bound for mutated genes
            high (float): higher bound for mutated genes

        Returns:
            new_genomes (np.array((N, d), float)): mutated genomes

        """
        if low > high:
            low, high = high, low

        mut_power = 1.0 / (self.eta + 1.)

        rands = np.random.rand(*g.shape)
        mask = rands < 0.5

        xy = np.full_like(g, np.nan)
        xy[mask] = 1.0 - ((g[mask] - low) / (high - low))
        xy[~mask] = 1.0 - ((high - g[~mask]) / (high - low))

        val = np.full_like(g, np.nan)
        val[mask] = 2.0 * rands[mask] + (1.0 - 2.0 * rands[mask]) * xy[mask] ** (self.eta + 1)
        val[~mask] = 2.0 * (1.0 - rands[~mask]) + 2.0 * (rands[~mask] - 0.5) * xy[~mask] ** (self.eta + 1)

        delta_q = np.full_like(g, np.nan)
        delta_q[mask] = val[mask] ** mut_power - 1.0
        delta_q[~mask] = 1.0 - val[~mask] ** mut_power

        new_genomes = g + delta_q * (high - low)
        new_genomes = np.minimum(np.maximum(new_genomes, low * np.ones_like(new_genomes)), high * np.ones_like(new_genomes))

        return new_genomes

    def compute_novelty(self, new_b, old_b=None):
        """Compute the novelty of new behaviors, compared to a pool of new + old behaviors.

        Different strategies are possible to compute the novelty:
        - "novelty": novelty is the average distance to the n_neighbors closest neighbors
        - "hull": novelty is the smallest distance to the hull of old_b ([min(old_b), max(old_b)])
        - "fitness": novelty is the distance to the best_fit target
        - "random": novelty is randomly assigned from a unitary uniform distribution

        Args:
            new_b (np.array((N), float)): behaviors to compute the novelty for
            old_b (np.array((N), float)): archive of behaviors used as reference to compute the novelty

        Returns:
            novelties (np.array((N), float)): novelties of the new behaviors

        """
        if self.criterion not in ["novelty", "hull", "fitness", "random"]:
            raise ValueError("criterion can only be 'novelty', 'hull', 'fitness', or 'random'")

        if self.criterion == "novelty":
            if old_b is None:
                old_b = self.get_reference_behaviors() # get the behavior of archive, population, and offspring at step t
            # distances = cdist(new_b.reshape(-1, 1), old_b.reshape(-1, 1))
            distances = cdist(new_b, old_b)
            distances = np.sort(distances, axis=1)
            novelties = np.mean(distances[:, :self.n_neighbors], axis=1)  # Note: one distance might be 0 (distance to self)

        elif self.criterion == "hull":
            if old_b is None:
                old_b = self.get_reference_behaviors()
            hull_min = np.min(old_b)
            hull_max = np.max(old_b)
            smallest_distances = np.maximum(hull_min - new_b, new_b - hull_max)
            novelties = np.maximum(smallest_distances, np.zeros_like(smallest_distances))

        elif self.criterion == "fitness":
            novelties = -np.abs(new_b - self.best_fit)  # distance to the best fit

        elif self.criterion == "random":
            novelties = np.random.rand(len(new_b))

        else:
            novelties = []

        return novelties

  

    def initialize_population(self):
        """Generates an initial population of individuals at generation self.t.

        """
        torch.manual_seed(0)
        X_original = torch.rand(2000,dim_latent)
        self.population[self.t, :, self.I_PARENT] = np.full(self.n_pop, np.nan)
        self.population[self.t, :, self.I_GENERATION] = -np.ones(self.n_pop, dtype=np.float32)
        self.population[self.t, :, self.I_SELECTED_POP] = np.zeros(self.n_pop, dtype=np.float32)
        self.population[self.t, :, self.I_SELECTED_ARC] = np.zeros(self.n_pop, dtype=np.float32)
        self.population[self.t, :, self.I_AGE] = np.zeros(self.n_pop, dtype=np.float32)
        
        # self.population[self.t, :, :][:, I_GENOME] = np.random.rand(self.n_pop, 2) - 0.5
        np.random.seed(self.seed)
        ids_acquired = np.random.choice(np.arange((len(X_original))), size=self.n_pop, replace=False)
        # self.population[self.t, :, :][:, self.I_GENOME] = np.random.rand(self.n_pop, self.dim)
        self.population[self.t, :, :][:, self.I_GENOME] = X_original[ids_acquired]
        
        # sobol = SobolEngine(dimension=self.dim, scramble=True,seed=self.seed)
        # self.population[self.t, :, :][:, self.I_GENOME] = np.array(sobol.draw(n=self.n_pop).to(torch.float64))
        
        # self.population[self.t, :, :][:, self.I_GENOME] = self.lb + (self.ub-self.lb)*np.random.rand(self.n_pop, self.dim)
        self.population[self.t, :, self.I_BEHAVIOR: self.I_BEHAVIOR+196] = self.gene_to_behavior(self.population[self.t, :, :][:, self.I_GENOME])
        self.population[self.t, :, self.I_DISTANCE] = np.zeros(self.n_pop, dtype=np.float32)
        self.population[self.t, :, self.I_NOVELTY] = self.compute_novelty(self.population[self.t, :, self.I_BEHAVIOR:self.I_BEHAVIOR+196],
                                                                     old_b=self.population[self.t, :, self.I_BEHAVIOR:self.I_BEHAVIOR+196])  # compare only to itself
        # self.population[self.t, :, self.I_COVERAGE], self.population[self.t, :, self.I_UNIFORMITY],\
        #     self.population[self.t, :, self.I_CUM_COVERAGE], self.population[self.t, :, self.I_CUM_UNIFORMITY] = \
        #     self.evaluate_coverage_and_uniformity(self.population[self.t, :, :][:, self.I_GENOME])

    def initialize_archive(self):
        """Add most novel individuals for the generation self.t to the archive.

        """
        # add the best individuals to the archive
        selected, self.archive[self.t, :, :] = self.select_individuals(self.population[self.t, :, :], self.n_selected, "novelty")
        self.population[self.t, selected, self.I_SELECTED_ARC] = 1.

    def generate_offsprings(self):
        """Generates offsprings from the population at generation self.t.

        """
        self.offsprings[self.t, :, self.I_GENERATION] = float(self.t)
        self.offsprings[self.t, :, self.I_PARENT] = np.random.randint(0, self.n_pop, self.n_offspring)
        self.offsprings[self.t, :, self.I_SELECTED_POP] = np.zeros(self.n_offspring, dtype=np.float32)
        self.offsprings[self.t, :, self.I_SELECTED_ARC] = np.zeros(self.n_offspring, dtype=np.float32)
        self.offsprings[self.t, :, self.I_AGE] = np.zeros(self.n_offspring, dtype=np.float32)
        self.offsprings[self.t, :, :][:, self.I_GENOME] = \
            self.mutate_genome(self.population[self.t, self.offsprings[self.t, :, self.I_PARENT].astype(int), :][:, self.I_GENOME], low=0, high=1)
        self.offsprings[self.t, :, self.I_BEHAVIOR:self.I_BEHAVIOR+196] = self.gene_to_behavior(self.offsprings[self.t, :, :][:, self.I_GENOME])
        # self.offsprings[self.t, :, self.I_DISTANCE] = np.abs(self.offsprings[self.t, :, self.I_BEHAVIOR]
        #                                                 - self.population[self.t, self.offsprings[self.t, :, self.I_PARENT].astype(int), self.I_BEHAVIOR])
        old_behaviors = self.get_reference_behaviors()
        self.offsprings[self.t, :, self.I_NOVELTY] = self.compute_novelty(self.offsprings[self.t, :, self.I_BEHAVIOR:self.I_BEHAVIOR+196], old_behaviors)
        # self.offsprings[self.t, :, self.I_COVERAGE], self.offsprings[self.t, :, self.I_UNIFORMITY],\
        #     self.offsprings[self.t, :, self.I_CUM_COVERAGE], self.offsprings[self.t, :, self.I_CUM_UNIFORMITY] = \
        #     self.evaluate_coverage_and_uniformity(self.offsprings[self.t, :, :][:, self.I_GENOME])

    # @staticmethod
    def select_individuals(self,individuals, n, strategy):
        """Selects n individuals according to the strategy.

        Args:
            individuals (np.array((N, d), float)): set of individuals
            n (in): number of individuals to select
            strategy (str): strategy to select the individuals - can be 'novelty' or 'random'

        Returns:
            selected (list(int)): index of the selected individuals
            (np.array((n, d), float)): set of selected individuals

        """
        if strategy == "novelty":
            selected = np.argsort(individuals[:, self.I_NOVELTY])[-n:]
        elif strategy == "random":
            selected = np.random.choice(individuals.shape[0], n, replace=False)
        else:
            selected = []
        return selected, individuals[selected, :]

    def select_and_add_to_archive(self):
        selected, self.archive[self.t + 1, :, :] = self.select_individuals(self.offsprings[self.t, :, :], self.n_selected, self.addition)
        self.offsprings[self.t, selected, self.I_SELECTED_ARC] = 1.

    def get_reference_behaviors(self):
        """Concatenates the behaviors from the archive, population, and offsprings from generation t or the frozen generation.

        Returns:
            (np.array((N,d), float)): set of behaviors

        """
        return np.vstack((self.population[min(self.t, self.frozen), :, self.I_BEHAVIOR:self.I_BEHAVIOR+196],
                          self.offsprings[min(self.t, self.frozen), :, self.I_BEHAVIOR:self.I_BEHAVIOR+196],
                          self.archive[:(min(self.t, self.frozen) + 1), :, self.I_BEHAVIOR:self.I_BEHAVIOR+196][0]))

    def create_next_generation(self):
        extended_population = np.vstack((self.population[self.t, :, :], self.offsprings[self.t, :, :]))
        selected, self.population[self.t + 1, :, :] = self.select_individuals(extended_population, self.n_pop, "novelty") # select n_pop individuals from extended_polulation
        selected_population = selected[selected < self.n_pop]
        selected_offsprings = selected[selected >= self.n_pop] - self.n_pop
        self.population[self.t, selected_population, self.I_SELECTED_POP] = 1.
        self.offsprings[self.t, selected_offsprings, self.I_SELECTED_POP] = 1.
        self.population[self.t + 1, :, self.I_AGE] += 1.
        self.population[self.t + 1, :, self.I_SELECTED_POP] = 0.  # erase possible heritage from previous generation
        self.population[self.t + 1, :, self.I_SELECTED_ARC] = 0.  # erase possible heritage from previous generation
        # _, _, self.population[self.t + 1, :, self.I_CUM_COVERAGE], self.population[self.t + 1, :, self.I_CUM_UNIFORMITY] = \
        #     self.evaluate_coverage_and_uniformity(self.population[self.t + 1, :, :][:, self.I_GENOME])  # update the cumulative coverage and uniformity
    
    def evaluate_coverage_and_uniformity_custom(self, g):
        """Evaluate the coverage and uniformity of genome(s) via sampling.

        Args:
            g (np.array((N, d), float)): genomes
            n_bins (int): number of bins in the behavior space

        Returns:
            coverages (np.array((N), float)): ratios of bins covered by sampling each genomes
            uniformities (np.array((N), float)): uniformities of the sampling from each genomes
            cum_coverage (float): ratios of bins covered by sampling all genomes
            cum_uniformity (float): uniformity of the sampling from all genomes

        """
        num = g.shape[0]
        all_behavior_samples = g[:, self.I_BEHAVIOR:self.I_BEHAVIOR+196]

        # cumulative over all genomes
        cum_hist, _ = np.histogram(all_behavior_samples, np.linspace(self.obj_lb, self.obj_ub, self.n_bins + 1))
        cum_hist = cum_hist[np.nonzero(cum_hist)] / (num) # discrete distribution
        cum_hist_uni = np.mean(cum_hist) * np.ones_like(cum_hist) # theoretical uniform distribution

        cum_coverage = len(cum_hist) / self.n_bins
        cum_uniformity = 1 - jensenshannon(cum_hist, cum_hist_uni, base=2)
        
        cumbent = float(max(all_behavior_samples))
        return cum_coverage, cum_uniformity, cumbent
    
    def calculate_matric(self):
        all_population = self.population[0, :, :] # initial population
        all_population = all_population.reshape(-1, all_population.shape[-1])
        
        all_offspring = self.offsprings[0:(self.t+1), :, :] # all generated offspring
        all_offspring = all_offspring.reshape(-1, all_offspring.shape[-1])
        extended_population =  np.vstack((all_population, all_offspring))   
        # coverage, uniformity, cumbent = self.evaluate_coverage_and_uniformity_custom(extended_population)
        all_behavior_samples = all_population[:, self.I_BEHAVIOR:self.I_BEHAVIOR+196]
        test_output, last_layer = cnn_trained(torch.tensor(all_behavior_samples).cpu().view(len(all_behavior_samples),14,14).unsqueeze(1))
        # pred_y = torch.max(test_output, 1)[1].data.numpy().squeeze()
        pred_y = torch.max(test_output, 1).indices[torch.max(test_output, 1).values>0.99]
        coverage = len(np.unique(pred_y))
        
        self.extended_population = extended_population
        # cumbent = float(max(extended_population))
        cost = self.n_offspring * (self.t + 1)
        return coverage, cost
    
    def calculate_matric_initialize(self):
        all_population = self.population[0, :, :] # initial population
        all_population = all_population.reshape(-1, all_population.shape[-1])
        all_behavior_samples = all_population[:, self.I_BEHAVIOR:self.I_BEHAVIOR+196]
        test_output, last_layer = cnn_trained(torch.tensor(all_behavior_samples).cpu().view(len(all_behavior_samples),14,14).unsqueeze(1))
        # pred_y = torch.max(test_output, 1)[1].data.numpy().squeeze()
        pred_y = torch.max(test_output, 1).indices[torch.max(test_output, 1).values>0.99]
        coverage = len(np.unique(pred_y))
        # coverage, uniformity, cumbent = self.evaluate_coverage_and_uniformity_custom(all_population)
        # cumbent = float(max(all_population))
        return coverage
        
    def run_novelty_search(self):
        """Applies the Novelty Search algorithm."""

        # initialize the run - create a random population and add the best individuals to the archive
        self.initialize_population()
        self.initialize_archive()
        coverage = self.calculate_matric_initialize()
        coverage_list=[coverage]
        # uniformity_list=[uniformity]
        # cumbent_list = [cumbent]

        cost_list = [0]
        # iterate
        while self.t < self.n_epochs:

            # in case of a restart, reinitialize the population and start again
            if self.t == self.restart:
                self.initialize_population()

            # generate offsprings from random parents in the population
            self.generate_offsprings() # use mutation to generate offspring

            # update the population novelty
            self.population[self.t, :, self.I_NOVELTY] = self.compute_novelty(self.population[self.t, :, self.I_BEHAVIOR:self.I_BEHAVIOR+196])

            # add the most novel offsprings to the archive
            self.select_and_add_to_archive()

            # calculate the reachability and uniformity for all genomes (sampled data)
            coverage, cost = self.calculate_matric()
            coverage_list.append(coverage)
           
            cost_list.append(cost)
            # torch.save(torch.tensor(torch.tensor(self.extended_population[:, self.I_GENOME])), 'train_x_DEA_seed'+str(self.seed)+'.pt')
            # torch.save(torch.tensor(torch.tensor(self.extended_population[:, self.I_BEHAVIOR])), 'train_y_DEA_seed'+str(self.seed)+'.pt')
            # keep the most novel individuals in the (population + offsprings) in the population
            self.create_next_generation()

            # if self.display:
            #     self.display_generation()

            self.t += 1
            
        return coverage_list,  cost_list



    def get_results(self):
        """Yields the archive history, population history, and offsprings history in a dictionary.

        Returns:
            d (dict)

        """
        d = {"archive": self.archive,
             "population": self.population,
             "offsprings": self.offsprings}
        return d


def create_and_run_experiment(params, display=False, seed=None,dim=2, obj_lb = -5, obj_ub = 5, lb=-5, ub=5):
    """Creates a Novelty Search algorithm and run the search according to the input parameters.
       It is possible to display the evolution of the search.

    Args:
        params (dict): parameters of the search
        display (bool): flag to display each generation during the run

    Returns:
        data (dict): dictionary containing the archive history, population history, and offsprings history

    """
    my_exp = Experiment(params, display, seed, dim, obj_lb, obj_ub, lb, ub)
    coverage_list, cost_list = my_exp.run_novelty_search()
    # data = my_exp.get_results()
    return coverage_list,cost_list

def run_sequentially(params,seed,dim,obj_lb,obj_ub,lb,ub):
    coverage_list,cost_list = create_and_run_experiment(params,True,seed,dim,obj_lb,obj_ub,lb,ub)
    return coverage_list, cost_list
    
    



