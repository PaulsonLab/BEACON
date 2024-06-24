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
sys.path.append('/home/tang.1856/Jonathan/Hands-on-Neuroevolution-with-Python-master/Chapter6')
from maze_NS import Maze
from torch.quasirandom import SobolEngine
from scipy.spatial import ConvexHull

class Experiment:
    """
    Novelty search in a simple simulated setting where a 2D genome is mapped to a 1D behavior space via a simple  non-linear mapping.
    """

    def __init__(self, params, display=False, seed = None, dim = 2, obj_lb1 = -5, obj_ub1 = 5, obj_lb2 = -5, obj_ub2 = 5, lb = -5, ub = 5):
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
                                     '2D_cosinus', 'multiplicative_cosinus', 'peaks', '2D_peaks', 'Rosen', 'Ackley', 'Hartmann', 'StyTang','medium maze','Cluster'], "incorrect type of mapping"
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
        self.I_BEHAVIOR1 = 5+dim  # 1D behavior
        self.I_BEHAVIOR2 = 5+dim+1
        self.I_DISTANCE = 5+dim+2  # distance to the parent when the individual is created
        self.I_NOVELTY = 5+dim+3  # novelty of the individual
        self.I_COVERAGE = 5+dim+4  # coverage associated with the genome
        self.I_UNIFORMITY = 5+dim+5  # uniformity associated with the genome
        self.I_CUM_COVERAGE = 5+dim+6  # cumulative coverage associated with the current group of individuals
        self.I_CUM_UNIFORMITY = 5+dim+7  # cumulative uniformity associated with the current group of individuals
        self.SIZE_INDIVIDUAL = 5+dim+8  # length of the descriptor for each individual
        
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
        
        self.obj_lb1 = obj_lb1
        self.obj_ub1 = obj_ub1
        self.obj_lb2 = obj_lb2
        self.obj_ub2 = obj_ub2
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

        
        if self.mapping == 'Cluster':
            g = self.lb+(self.ub-self.lb)*g
            y1 = np.sin(g[:,0]) * np.cos(g[:,1]) + g[:,2] * np.exp(-g[:,0]**2) * np.cos(g[:,0] + g[:,1]) + 0.01*np.sin((g[:,3] + g[:,4] + g[:,5]))
            y2 = np.sin(g[:,3]) * np.cos(g[:,4]) + g[:,5] * np.exp(-g[:,3]**2) * np.sin(g[:,3] + g[:,4])+ 0.01*np.cos((g[:,0] + g[:,1] + g[:,2]))
            behavior = np.column_stack((y1,y2))
            
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
    
    def distance_point_to_segment(self,p, a, b):
        # Convert points to numpy arrays
        p, a, b = np.array(p), np.array(a), np.array(b)
        # Compute the projection of point p onto the line ab
        ap = p - a
        ab = b - a
        result = a + np.dot(ap, ab) / np.dot(ab, ab) * ab
        # Check if the projection is on the line segment
        if np.dot(result - a, result - b) > 0:
            # The projection is not on the segment, return the minimum distance to endpoints
            return min(np.linalg.norm(p - a), np.linalg.norm(p - b))
        else:
            # Return the perpendicular distance to the segment
            return np.linalg.norm(np.cross(ab, ap)) / np.linalg.norm(ab)
    
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
            distances = cdist(new_b, old_b)
            distances = np.sort(distances, axis=1)
            novelties = np.mean(distances[:, :self.n_neighbors], axis=1)  # Note: one distance might be 0 (distance to self)

        elif self.criterion == "hull":
            if old_b is None:
                old_b = self.get_reference_behaviors()
            hull = ConvexHull(old_b)  
            
            distances = []
            for q in range(len(new_b)):
                min_dist = float('inf')
                for simplex in hull.simplices:
                    
                    # new_points = np.append(old_b, new_b[0])
                    # new_hull = ConvexHull(new_points)
                    # if list(new_hull.vertices) == list(hull.vertices): # inside the convex hull
                    dist = self.distance_point_to_segment(new_b[q], old_b[simplex[0]], old_b[simplex[1]])
                    min_dist = min(min_dist, dist)
                distances.append(min_dist)
            
            novelties = np.array(distances)
            
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
        self.population[self.t, :, self.I_PARENT] = np.full(self.n_pop, np.nan)
        self.population[self.t, :, self.I_GENERATION] = -np.ones(self.n_pop, dtype=np.float32)
        self.population[self.t, :, self.I_SELECTED_POP] = np.zeros(self.n_pop, dtype=np.float32)
        self.population[self.t, :, self.I_SELECTED_ARC] = np.zeros(self.n_pop, dtype=np.float32)
        self.population[self.t, :, self.I_AGE] = np.zeros(self.n_pop, dtype=np.float32)
        
        # self.population[self.t, :, :][:, I_GENOME] = np.random.rand(self.n_pop, 2) - 0.5
        np.random.seed(self.seed)
        self.population[self.t, :, :][:, self.I_GENOME] = np.random.rand(self.n_pop, self.dim)
        
        # sobol = SobolEngine(dimension=self.dim, scramble=True,seed=self.seed)
        # self.population[self.t, :, :][:, self.I_GENOME] = np.array(sobol.draw(n=self.n_pop).to(torch.float64))
        
        # self.population[self.t, :, :][:, self.I_GENOME] = self.lb + (self.ub-self.lb)*np.random.rand(self.n_pop, self.dim)
        self.population[self.t, :, self.I_BEHAVIOR1:self.I_BEHAVIOR2+1] = self.gene_to_behavior(self.population[self.t, :, :][:, self.I_GENOME])
        
        

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
        self.offsprings[self.t, :, self.I_BEHAVIOR1:self.I_BEHAVIOR2+1] = self.gene_to_behavior(self.offsprings[self.t, :, :][:, self.I_GENOME])
        # self.offsprings[self.t, :, self.I_DISTANCE] = np.abs(self.offsprings[self.t, :, self.I_BEHAVIOR1:self.I_BEHAVIOR2]
        #                                                 - self.population[self.t, self.offsprings[self.t, :, self.I_PARENT].astype(int), self.I_BEHAVIOR])
        old_behaviors = self.get_reference_behaviors()
        self.offsprings[self.t, :, self.I_NOVELTY] = self.compute_novelty(self.offsprings[self.t, :, self.I_BEHAVIOR1:self.I_BEHAVIOR2+1], old_behaviors)
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
        return np.vstack((self.population[min(self.t, self.frozen), :, self.I_BEHAVIOR1:self.I_BEHAVIOR2+1],
                          self.offsprings[min(self.t, self.frozen), :, self.I_BEHAVIOR1:self.I_BEHAVIOR2+1],
                          self.archive[:(min(self.t, self.frozen) + 1), :, self.I_BEHAVIOR1:self.I_BEHAVIOR2+1][0]))

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
        all_behavior_samples = torch.tensor(g[:, self.I_BEHAVIOR1:self.I_BEHAVIOR2+1])
        
        cell_size1 = (self.obj_ub1 - self.obj_lb1) / self.n_bins
        cell_size2 = (self.obj_ub2 - self.obj_lb2) / self.n_bins
        
        indices_y1 = ((all_behavior_samples[:, 0] - self.obj_lb1) / cell_size1).floor().int()
        indices_y2 = ((all_behavior_samples[:, 1] - self.obj_lb2) / cell_size2).floor().int()
        grid_indices = torch.stack((indices_y1, indices_y2), dim=1)
        unique_cells = set(map(tuple, grid_indices.tolist()))
        return len(unique_cells)/(self.n_bins**2)
    
    def calculate_matric(self):
        all_population = self.population[0, :, :] # initial population
        all_population = all_population.reshape(-1, all_population.shape[-1])
        
        all_offspring = self.offsprings[0:(self.t+1), :, :] # all generated offspring
        all_offspring = all_offspring.reshape(-1, all_offspring.shape[-1])
        extended_population =  np.vstack((all_population, all_offspring))   
        coverage  = self.evaluate_coverage_and_uniformity_custom(extended_population)
        self.extended_population = extended_population
        # cumbent = float(max(extended_population))
        cost = self.n_offspring * (self.t + 1)
        return coverage, cost
    
    def calculate_matric_initialize(self):
        all_population = self.population[0, :, :] # initial population
        all_population = all_population.reshape(-1, all_population.shape[-1])
  
        coverage = self.evaluate_coverage_and_uniformity_custom(all_population)
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
            self.population[self.t, :, self.I_NOVELTY] = self.compute_novelty(self.population[self.t, :, self.I_BEHAVIOR1:self.I_BEHAVIOR2+1])

            # add the most novel offsprings to the archive
            self.select_and_add_to_archive()

            # calculate the reachability and uniformity for all genomes (sampled data)
            coverage, cost = self.calculate_matric()
            coverage_list.append(coverage)
            # uniformity_list.append(uniformity)
            # cumbent_list.append(cumbent)
            cost_list.append(cost)
            # torch.save(torch.tensor(torch.tensor(self.extended_population[:, self.I_GENOME])), 'train_x_DEA_seed'+str(self.seed)+'.pt')
            # torch.save(torch.tensor(torch.tensor(self.extended_population[:, self.I_BEHAVIOR])), 'train_y_DEA_seed'+str(self.seed)+'.pt')
            # keep the most novel individuals in the (population + offsprings) in the population
            self.create_next_generation()

            # if self.display:
            #     self.display_generation()

            self.t += 1
            
        return coverage_list, cost_list



    def get_results(self):
        """Yields the archive history, population history, and offsprings history in a dictionary.

        Returns:
            d (dict)

        """
        d = {"archive": self.archive,
             "population": self.population,
             "offsprings": self.offsprings}
        return d


def create_and_run_experiment(params, display=False, seed=None,dim=2, obj_lb1 = -5, obj_ub1 = 5, obj_lb2 = -5, obj_ub2 = 5, lb=-5, ub=5):
    """Creates a Novelty Search algorithm and run the search according to the input parameters.
       It is possible to display the evolution of the search.

    Args:
        params (dict): parameters of the search
        display (bool): flag to display each generation during the run

    Returns:
        data (dict): dictionary containing the archive history, population history, and offsprings history

    """
    my_exp = Experiment(params, display, seed, dim, obj_lb1, obj_ub1, obj_lb2, obj_ub2, lb, ub)
    coverage_list,  cost_list = my_exp.run_novelty_search()
    # data = my_exp.get_results()
    return coverage_list,  cost_list

def run_sequentially(params,seed,dim,obj_lb1,obj_ub1,obj_lb2,obj_ub2,lb,ub):
    coverage_list,  cost_list = create_and_run_experiment(params,True,seed,dim,obj_lb1,obj_ub1,obj_lb2,obj_ub2,lb,ub)
    return coverage_list, cost_list
    
    



