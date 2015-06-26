"""
Entropy Rate
"""

from collections import defaultdict
from numpy import log

def entropy_rate(edges, stationary):
    """
    Computes the entropy rate given the edges of the process and the stationary distribution.

    Parameters
    ----------
    edges: list of tuples
        Transition probabilities of the form [(source, target, transition_probability
    stationary: dictionary
        Precomputed stationary distribution

    Returns
    -------
    float, entropy rate of the process
    """

    e = defaultdict(float)
    for a,b,v in edges:
        e[a] -= stationary[a] * v * log(v)
    return sum(e.values())

def entropy_rate_func(N, edge_func, stationary):
    """
    Computes entropy rate for a process with a large transition matrix, defined
    by a transition function (edge_func) rather than a list of weighted edges.

    Use when the number of states or the transition matrix is prohibitively
    large, e.g. for the Wright-Fisher process.

    Parameters
    ----------
    N: int
        Population size / simplex divisor
    edge_func, function
        Yields the transition probabilities between two states, edge_func(a,b)
    stationary: dictionary
        Precomputed stationary distribution
    """

    e = defaultdict(float)
    for a in simplex_generator(N):
        for b in simplex_generator(N):
            v = edge_func(a,b)
            e[a] -= stationary[a] * v * log(v)
    return sum(e.values())