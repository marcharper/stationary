"""Entropy rate computation."""

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
        for a, b, v in edges:
            e[a] -= stationary[a] * v * log(v)
        return sum(e.values())
    elif isinstance(edges, Callable):
        if not states:
            raise ValueError(
                "Keyword argument `states` required with edge_func")
        for a in states:
            for b in states:
                v = edges(a, b)
                e[a] -= stationary[a] * v * log(v)
        return sum(e.values())
