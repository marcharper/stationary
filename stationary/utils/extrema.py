import math
import numpy

from .math_helpers import one_step_generator
from .graph import Graph


def find_local_minima(d, comp_func=None):
    """
    Finds local minima of distributions on the simplex.

    Parameters
    ----------
    d: dict
        The dictionary on a simplex discretization to find the extrema of
   comp_func: function
        Function to compare states

    Returns
    -------
    set of minimal states.
    """

    if not comp_func:
        comp_func = lambda x, y: (x - y >= 0)

    dim = len(list(d)[0]) - 1
    states = []
    for state, value in d.items():
        if value is None:
            continue
        if math.isnan(value):
            continue
        is_extremum = True
        for one_step in one_step_generator(dim):
            adj = tuple(numpy.array(state) + numpy.array(one_step))
            try:
                v2 = d[adj]
            except KeyError:
                continue
            if comp_func(value, v2):
                is_extremum = False
                break
        if is_extremum:
            states.append(state)
    return set(states)


def find_local_maxima(d):
    """
    Finds local maxima of distributions on the simplex.

    Parameters
    ----------
    d: dict
        The dictionary on a simplex discretization to find the extrema of

    Returns
    -------
    set of maximal states.
    """

    comp_func = lambda x, y: (y - x >= 0)
    return find_local_minima(d, comp_func=comp_func)


def inflow_outflow(edges):
    """
    Computes the inflow - outflow of probability at each state.
    """

    g = Graph(edges)

    flow = dict()
    for s1 in g.vertices():
        flow[s1] = sum(g.out_dict(s1).values()) - sum(g.in_dict(s1).values())
    return flow
