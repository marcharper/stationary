
from collections import defaultdict, Callable

import numpy

from math_helpers import normalize, q_divergence


def expected_divergence(edges, states=None, q_d=1, boundary=True):
    """
    Computes the KL-div of the expected state with the state, for all states.

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
    boundary: bool, False
        Exclude the boundary states

    Returns
    -------
    Dictionary mapping states to D(E(state), state)
    """

    dist = q_divergence(q_d)
    e = defaultdict(float)

    if isinstance(edges, list):
        for x, y, w in edges:
            e[x] += numpy.array(y) * w
    elif isinstance(edges, Callable):
        if not states:
            raise ValueError, "Keyword argument `states` required with edge_func"
        for x in states:
            e[x] = 0.
            for y in states:
                w = edges(x,y)
                e[x] += numpy.array(y) * w
    d = dict()
    for state, v in e.items():
        # Some divergences do not play well on the boundary
        if not boundary:
            p = 1.
            for s in state:
                p *= s
            if p == 0:
                continue
        d[state] = dist(normalize(v), normalize(list(state)))
    return d
