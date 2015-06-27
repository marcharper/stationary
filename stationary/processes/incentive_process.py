"""
Calculates transitions for the Moran process and generalizations.
"""

from collections import defaultdict

from ..utils.math_helpers import kl_divergence, simplex_generator, one_step_indicies_generator, dot_product, normalize, q_divergence, logsumexp, log_factorial, log_inc_factorial, factorial, inc_factorial
from ..utils.edges import edges_to_matrix, edge_func_to_edges, states_from_edges, power_transitions

import numpy
from numpy import array, log, exp
#from numpy.linalg import matrix_power

from incentives import *


## Moran/Incentive Process

def is_valid_state(state, lower, upper):
    """
    Checks the bounds of a state to make sure it is a valid popualation state.
    """

    for i in state:
        if i < lower or i > upper:
            return False
    return True

def multivariate_transitions(N, incentive, num_types=3, mu=0.001, no_boundary=False):
    """
    Computes transition probabilities the Incentive process

    Parameters
    ----------
    N: int
        Population size / simplex divisor
    incentive: function
        An incentive function from incentives.py
    num_types: int, 3
        Number of types in population
    mu: float, 0.001
        The mutation rate of the process
    no_boundary: bool, False
        Exclude the boundary states
    """

    return list(multivariate_transitions_gen(N, incentive, num_types=num_types, mu=mu, no_boundary=no_boundary))

def multivariate_transitions_gen(N, incentive, num_types=3, mu=0.001, no_boundary=False):
    """
    Computes transition probabilities the Incentive process (generator),

    Parameters
    ----------
    N: int
        Population size / simplex divisor
    incentive: function
        An incentive function from incentives.py
    num_types: int, 3
        Number of types in population
    mu: float, 0.001
        The mutation rate of the process
    no_boundary: bool, False
        Exclude the boundary states
    """

    d = num_types - 1
    edges = []
    one_step_indicies = list(one_step_indicies_generator(d))
    if no_boundary:
        lower, upper = 1, N-1
    else:
        lower, upper = 0, N
    for state in simplex_generator(N, d):
        if no_boundary:
            is_boundary = False
            for i in state:
                if i == 0:
                    is_boundary = True
                    break
            if is_boundary:
                continue
        s = 0.
        inc = incentive(state)
        denom = float(sum(inc))
        # Transition probabilities for each adjacent state.
        for plus_index, minus_index in one_step_indicies:
            target_state = list(state)
            target_state[plus_index] += 1
            target_state[minus_index] -= 1
            target_state = tuple(target_state)
            # Is this a valid state? I.E. Are we on or near the boundary?
            if not is_valid_state(target_state, lower, upper):
                continue
            #mutations = [mu] * num_types
            #mutations[plus_index] = 1. - d*mu
            mutations = [mu / d] * num_types
            mutations[plus_index] = 1. - mu
            r = dot_product(inc, mutations) / denom
            transition = r * state[minus_index] / float(N)
            yield (state, target_state, transition)
            s += transition
        # Add in the transition probability for staying put.
        yield (state, state, 1. - s)

# Deletion candidate
def log_multivariate_transitions(N, logincentive, num_types=3, mu=0.001, no_boundary=False):
    """
    Computes transition probabilities the Incentive process in log-space

    Parameters
    ----------
    N: int
        Population size / simplex divisor
    incentive: function
        An incentive function from incentives.py
    num_types: int, 3
        Number of types in population
    mu: float, 0.001
        The mutation rate of the process
    no_boundary: bool, False
        Exclude the boundary states
    """

    d = num_types - 1
    edges = []
    one_step_indicies = list(one_step_indicies_generator(d))
    if no_boundary:
        lower, upper = 1, N-1
    else:
        lower, upper = 0, N
    for state in simplex_generator(N, d):
        if no_boundary:
            is_boundary = False
            for i in state:
                if i == 0:
                    is_boundary = True
                    break
            if is_boundary:
                continue
        inc = logincentive(state)
        denom = logsumexp(inc)
        # Transition probabilities for each adjacent state.
        logtransitions = []
        for plus_index, minus_index in one_step_indicies:
            target_state = list(state)
            target_state[plus_index] += 1
            target_state[minus_index] -= 1
            target_state = tuple(target_state)
            # Is this a valid state? I.E. Are we on or near the boundary?
            if not is_valid_state(target_state, lower, upper):
                continue
            mutations = [mu / d] * num_types
            mutations[plus_index] = 1. - mu
            r = logsumexp(inc, b=mutations) - denom
            logtransition = r + log(state[minus_index]) - log(N)
            edges.append((state, target_state, logtransition))
            logtransitions.append(logtransition)
        edges.append((state, state, log(1.-exp(logsumexp(logtransitions)))))
    return edges

def compute_edges(N=30, num_types=2, m=None, incentive_func=logit, beta=1., q=1., mu=None):
    """
    Wrapper function of multivariate_transitions with some reasonable defaults.
    """
    if not m:
        m = numpy.ones((num_types, num_types)) # neutral landscape
    if not num_types:
        num_types = len(m[0])
    fitness_landscape = linear_fitness_landscape(m)
    incentive = incentive_func(fitness_landscape, beta=beta, q=q)
    if not mu:
        # mu = (n-1.)/n * 1./(N+1) # Match with Traulsen's form
        mu = 1./N
    edges = multivariate_transitions(N, incentive, num_types=num_types, mu=mu)
    return edges

#def kl(edges, q_d=1, boundary=True):
    #"""
    #Computes the KL-div of the expected state with the state, for all states.

    #Parameters
    #----------
    #edges: list of tuples
        #Transition probabilities of the form [(source, target, transition_probability
    #q_d: float, 1
        #parameter that specifies which divergence function to use
    #boundary: bool, False
        #Exclude the boundary states

    #Returns
    #-------
    #Dictionary mapping states to D(E(state), state)
    #"""

    #N = sum(edges[0][0])
    #dist = q_divergence(q_d)
    #e = defaultdict(float)
    #for x, y, w in edges:
        #e[x] += numpy.array(y) * w
    #d = dict()
    #for state, v in e.items():
        ## KL doesn't play well on the boundary.
        #if not boundary:
            #p = 1.
            #for s in state:
                #p *= s
            #if p == 0:
                #continue
        #d[state] = dist(normalize(v), normalize(list(state)))
    #return d

## k-fold Moran process

def k_fold_incentive_transitions(N, incentive, num_types, mu=None, k=None):
    """
    Computes transition probabilities the k-fold incentive process

    Parameters
    ----------
    N: int
        Population size / simplex divisor
    incentive: function
        An incentive function from incentives.py
    num_types: int, 3
        Number of types in population
    mu: float, 0.001
        The mutation rate of the process
    k: int, N // 2
        The power of the process
    """

    if not k:
        k = N // 2
    if not mu:
        mu = 1. / N
    edges = multivariate_transitions(N, incentive, num_types=num_types, mu=mu)
    edge_func = power_transitions(edges, k)
    states = states_from_edges(edges)
    new_edges = edge_func_to_edges(edge_func, states)
    return new_edges

## Neutral landscape / Dirichlet

def neutral_stationary(N, alpha, n=3, logspace=False):
    """
    Computes the stationary distribution of the neutral landscape. This process
    is always reversible and there is an explicit formula.

    Parameters
    ----------
    N: int
        Population size / simplex divisor
    alpha:
        Parameter defining the stationary distribution in terms of n and mu
    n: int, 3
        Simplex dimension - 1, Number of types in population
    logspace: bool, False
        Use the logspace version

    Returns
    -------
    dictionary, stationary distribution of the process
    """

    # Large N is better handled by the log version to avoid underflows
    if logspace or (N > 100):
        return log_neutral_stationary(N, alpha, n=n)

    # Just compute the distribution directly for each state
    d2 = dict()
    for state in simplex_generator(N, n-1):
        t = 1.
        for i in state:
            t *= inc_factorial(alpha, i) / factorial(i)
        t *= factorial(N) / inc_factorial(n * alpha, N)        
        d2[state] = t
    return d2

def log_neutral_stationary(N, alpha, n=3):
    """
    Computes the stationary distribution of the neutral landscape. This process
    is always reversible and there is an explicit formula. This function is the 
    same as neutral_stationary in log-space.

    Parameters
    ----------
    N: int
        Population size / simplex divisor
    alpha:
        Parameter defining the stationary distribution in terms of n and mu
    n: int, 3
        Simplex dimension - 1, Number of types in population

    Returns
    -------
    dictionary, stationary distribution of the process
    """

    d2 = dict()
    for state in simplex_generator(N, n-1):
        t = 0.
        for i in state:
            t += log_inc_factorial(alpha, i) - log_factorial(i)
        t += log_factorial(N) - log_inc_factorial(n * alpha, N)
        d2[state] = exp(t)
    return d2
