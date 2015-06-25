import numpy

from math_helpers import kl_divergence_dict
from graph import Graph
from edges import edges_to_edge_dict

from nose.tools import assert_almost_equal, assert_equal, assert_raises, \
                       assert_true

def check_detailed_balance(edges, s, places=7):
    """
    Check if the detailed balance condition is satisfied.

    Parameters
    ----------
    edges: list of tuples
        transitions of the Markov process
    s: dict
        the stationary distribution
    """

    edge_dict = edges_to_edge_dict(edges)
    for s1, s2 in edge_dict.keys():
            diff = s[s1] * edge_dict[(s1, s2)] - s[s2] *edge_dict[(s2, s1)]
            assert_almost_equal(diff, 0, places=places)

def check_global_balance(edges, stationary, places=7):
    """
    Checks that the stationary distribution satisfies the global balance
    condition. https://en.wikipedia.org/wiki/Balance_equation

    Parameters
    ----------
    edges: list of tuples
        transitions of the Markov process
    s: dict
        the stationary distribution
    """

    g = Graph(edges)

    for s1 in g.vertices():
        lhs = 0.
        rhs = 0.
        for s2, v in g.out_dict(s1).items():
            if s1 == s2:
                continue
            lhs += stationary[s1] * v
        for s2, v in g.in_dict(s1).items():
            if s1 == s2:
                continue
            rhs += stationary[s2] * v
        assert_almost_equal(lhs, rhs, places=places)

def check_eigenvalue(edges, s, places=3):
    """
    Check that the stationary distribution satisfies the eigenvalue condition.

    Parameters
    ----------
    edges: list of tuples
        transitions of the Markov process
    s: dict
        the stationary distribution
    """

    g = Graph(edges)
    t = g.left_multiply(s)
    assert_almost_equal(kl_divergence_dict(s, t), 0, places=places)
