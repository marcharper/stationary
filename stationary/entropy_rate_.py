"""
Entropy Rate
"""

from collections import defaultdict, Callable
from numpy import log

def entropy_rate(edges, stationary, states=None):
    """
    Computes the entropy rate given the edges of the process and the stationary distribution.

    Parameters
    ----------
    edges: list of tuples or function
        The transitions of the process, either a list of (source, target,
        transition_probability), or an edge_function that takes two parameters,
        the source and target states, to the transition transition probability.
        If using an edge_function you must supply the states of the process.
    q_d: float, 1
        parameter that specifies which divergence function to use
    states: list, None
        States for use with the edge_func
    stationary: dictionary
        Precomputed stationary distribution

    Returns
    -------
    float, entropy rate of the process
    """

    e = defaultdict(float)
    if isinstance(edges, list):
        for a,b,v in edges:
            e[a] -= stationary[a] * v * log(v)
        return sum(e.values())
    elif isinstance(edges, Callable):
        if not states:
            raise ValueError, "Keyword argument `states` required with edge_func"
        for a in states:
            for b in states:
                v = edges(a,b)
                e[a] -= stationary[a] * v * log(v)
        return sum(e.values())

#def entropy_rate_func(N, edge_func, stationary):
    #"""
    #Computes entropy rate for a process with a large transition matrix, defined
    #by a transition function (edge_func) rather than a list of weighted edges.

    #Use when the number of states or the transition matrix is prohibitively
    #large, e.g. for the Wright-Fisher process.

    #Parameters
    #----------
    #N: int
        #Population size / simplex divisor
    #edge_func, function
        #Yields the transition probabilities between two states, edge_func(a,b)
    #stationary: dictionary
        #Precomputed stationary distribution
    #"""

    #e = defaultdict(float)
    #for a in simplex_generator(N):
        #for b in simplex_generator(N):
            #v = edge_func(a,b)
            #e[a] -= stationary[a] * v * log(v)
    #return sum(e.values())
