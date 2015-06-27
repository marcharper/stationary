
from collections import defaultdict, Callable

import numpy

from math_helpers import normalize, q_divergence


def expected_divergence(edges, states=None, q_d=1, boundary=True):
    """
    Computes the KL-div of the expected state with the state, for all states.

    Parameters
    ----------
    edges: list of tuples or function
        Transition probabilities of the form [(source, target, transition_probability
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

#def expected_div_func(edge_func, states, q_d=1):
    #"""
    #Computes the KL-div of the expected state with the state, for all states.

    #Parameters
    #----------
    #edge_func, function
        #Yields the transition probabilities between two states, edge_func(a,b)
    #q_d: float, 1
        #parameter that specifies which divergence function to use

    #Returns
    #-------
    #Dictionary mapping states to D(E(state), state)
    #"""

    #e = dict()
    #dist = q_divergence(q_d)
    #for x in states:
        #e[x] = 0.
        #for y in states:
            #w = edge_func(x,y)
            ## Could use the fact that E(x) = n p here instead for efficiency, but this is relatively fast compared to the stationary calculation already, and is a good check of the identity. This would require a rewrite using the transition functions.
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