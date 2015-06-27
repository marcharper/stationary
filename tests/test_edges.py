
from nose.tools import assert_almost_equal, assert_equal, assert_raises, assert_true, assert_less_equal, assert_greater_equal, assert_greater

from stationary.utils.edges import *


def test_edge_functions():
    """
    """

    edges = [(0, 0, 1./3), (0, 1, 1./3), (0, 2, 1./3),
             (1, 0, 1./4), (1, 1, 1./2), (1, 2, 1./4),
             (2, 0, 1./6), (2, 1, 1./3), (2, 2, 1./2),]
    edges.sort()

    states = states_from_edges(edges)
    assert_equal(states, set([0, 1, 2]))
