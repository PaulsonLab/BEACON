# Synthetic Problem
parameters = {
    "mapping": "Ackley",                   # type of genome-to-behavior mapping {'linear', 'hyperbolic', 'bounded_linear', 'linear_seesaw', 'multiplicative', 'soft_multiplicative', 'hyperbolic_seesaw', 'multiplicative_seesaw', 'cosinus', '2D_cosinus', 'multiplicative_cosinus', 'peaks', '2D_peaks'}
    "eta": 45,                                 # mutation spread parameter
    "n_pop": 50,                               # number of individuals in the population (20)
    "n_offspring": 10,                         # number of offsprings generated from the population at each generation (40)
    "criterion": "novelty",                    # type of novelty computation {'novelty', 'hull', 'fitness', 'random'}
    "n_neighbors": 10,                         # number of closest neighbors to compute the "novelty" criterion
    "best_fit": -4,                            # arbitrary behavior with the maximum fitness for the "fitness" criterion
    "n_selected": 6,                           # number of offsprings added to the archive at each generation
    "addition": "novelty",                      # strategy to add individuals to the archive {'novelty', 'random'}
    "n_evolvability": 1000,                    # number of samples generated from each genome to evaluate its evolvability
    "n_epochs": 50,                          # number of generations of the search process
    "restart": 99999,                          # generation at which the population is re-initialized
    "frozen": 99999,                           # generation at which the reference set for the novelty computation is frozen
    "n_runs": 100,                             # number of experiments to run in parallel
    "name": "standard_novelty_random_100runs",  # name of the results file (can be None))
    "n_bins":50,
    'noise':8
}

#Maze Problem
# parameters = {
#     "mapping": "medium maze",                   # type of genome-to-behavior mapping {'linear', 'hyperbolic', 'bounded_linear', 'linear_seesaw', 'multiplicative', 'soft_multiplicative', 'hyperbolic_seesaw', 'multiplicative_seesaw', 'cosinus', '2D_cosinus', 'multiplicative_cosinus', 'peaks', '2D_peaks'}
#     "eta": 45,                                 # mutation spread parameter 45
#     "n_pop": 30,                               # number of individuals in the population (20)
#     "n_offspring": 10,                         # number of offsprings generated from the population at each generation (40)
#     "criterion": "hull",                    # type of novelty computation {'novelty', 'hull', 'fitness', 'random'}
#     "n_neighbors": 10,                         # number of closest neighbors to compute the "novelty" criterion
#     "best_fit": -4,                            # arbitrary behavior with the maximum fitness for the "fitness" criterion
#     "n_selected": 6,                           # number of offsprings added to the archive at each generation
#     "addition": "novelty",                      # strategy to add individuals to the archive {'novelty', 'random'}
#     "n_evolvability": 1000,                    # number of samples generated from each genome to evaluate its evolvability
#     "n_epochs": 30,                          # number of generations of the search process
#     "restart": 99999,                          # generation at which the population is re-initialized
#     "frozen": 99999,                           # generation at which the reference set for the novelty computation is frozen
#     "n_runs": 100,                             # number of experiments to run in parallel
#     "name": "standard_novelty_random_100runs",  # name of the results file (can be None))
#     "n_bins":25
# }
