"""
Convenience functions for common situations.
"""

from stationary import stationary_distribution
from stationary.processes import incentive_process
from stationary.processes.incentives import replicator, linear_fitness_landscape
from stationary.utils.math_helpers import simplex_generator

from .entropy_rate_ import entropy_rate


def moran(N, game_matrix=None, mu=None, incentive_func=replicator,
          exact=False, logspace=False):
    """
    A convenience function for the Moran process with mutation. Computes the
    transition probabilities and the stationary distribution.

    The number of types is determined from the dimensions of the game_matrix.

    Parameters
    ----------
    N: int
        The population size
    game_matrix: list of lists or numpy matrix, None
        The game matrix of the process, e.g. [[1, 2], [2, 1]] for the two-type
        Hawk-Dove game. If not specified, the 2-type neutral landscape is used.
    mu: float, None
        The mutation rate, if None then `mu` is set to 1 / N
    incentive_func: function, replicator
        A function defining the process, e.g. the Moran process, logit, Fermi, etc.
        Incentives functions are in stationary.processes.incentives
    exact: bool, False
        Use the approximate or exact calculation function
    logspace: bool, False
        Compute in log-space or not

    Returns
    -------
    edges, s, er: the list of transitions, the stationary distribution, and the
    entropy rate.
    """

    if not game_matrix:
        game_matrix = [[1, 1], [1, 1]]
    if not mu:
        mu = 1. / N
    num_types = len(game_matrix[0])

    fitness_landscape = linear_fitness_landscape(game_matrix)
    incentive = incentive_func(fitness_landscape)
    edges = incentive_process.multivariate_transitions(
        N, incentive, num_types=num_types, mu=mu)
    s = stationary_distribution(edges, exact=exact, logspace=logspace)
    er = entropy_rate(edges, s)
    return edges, s, er


def wright_fisher(N, game_matrix=None, mu=None, incentive_func=replicator,
                  logspace=False):
    """
    A convenience function for the Moran process with mutation. Computes the
    transition probabilities and the stationary distribution.

    The number of types is determined from the dimensions of the game_matrix.

    Parameters
    ----------
    N: int
        The population size
    game_matrix: list of lists or numpy matrix, None
        The game matrix of the process, e.g. [[1, 2], [2, 1]] for the two-type
        Hawk-Dove game. If not specified, the 2-type neutral landscape is used.
    mu: float, None
        The mutation rate, if None then `mu` is set to 1 / N
    incentive_func: function, replicator
        A function defining the process, e.g. the Moran process, logit, Fermi,
        Incentives functions are in stationary.processes.incentives
    logspace: bool, False
        Compute in log-space or not

    Returns
    -------
    edges, s, er: the list of transitions, the stationary distribution, and the
    entropy rate.
    """

    if not game_matrix:
        game_matrix = [[1, 1], [1, 1]]
    if not mu:
        mu = 1. / N
    num_types = len(game_matrix[0])

    fitness_landscape = linear_fitness_landscape(game_matrix)
    incentive = incentive_func(fitness_landscape)
    edge_func = wright_fisher.multivariate_transitions(N, incentive, mu=mu,
                                                       num_types=num_types)
    states = list(simplex_generator(N, d=num_types-1))
    s = stationary_distribution(edge_func, states=states, iterations=4*N,
                                logspace=logspace)
    er = entropy_rate(edge_func, s)
    return edge_func, s, er
