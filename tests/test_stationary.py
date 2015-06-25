from __future__ import absolute_import

import numpy

from nose.tools import assert_almost_equal, assert_equal, assert_raises, \
                       assert_true

from stationary.stationary import approximate_stationary_distribution, log_approximate_stationary_distribution, neutral_stationary, log_neutral_stationary, edges_to_edge_dict, edges_to_matrix, states_from_edges

from processes.incentive_process import compute_edges
from processes.incentives import replicator

from utils.matrix_checks import check_detailed_balance, check_global_balance, check_eigenvalue

def test_stationary(t1=0.4, t2=0.6):
    """
    Test the stationary distribution computations a simple Markov process.
    """

    edges = [(0, 1, t1), (0, 0, 1. - t1), (1, 0, t2), (1, 1, 1. - t2)]
    s_0 = 1./(1. + t1 / t2)
    exact_stationary = {0: s_0, 1: 1 - s_0}

    for func in [approximate_stationary_distribution, log_approximate_stationary_distribution]:
        approx_stationary = func(edges)
        for i in [0, 1]:
            assert_almost_equal(exact_stationary[i], approx_stationary[i])

    # Check that the stationary distribution satistifies balance conditions
    for s in [exact_stationary, approx_stationary]:
        check_detailed_balance(edges, s)
        check_global_balance(edges, s)
        check_eigenvalue(edges, s)

def test_neutral(N=100., lim=1e-14):
    """
    Compare stationary distribution computations to known analytic form for
    neutral landscape.
    """
    for n, N in [(2, 10), (2, 100), (3, 10), (3, 20), (4, 10)]:
        mu = (n-1.)/n * 1./(N+1)
        alpha = N * mu / (n - 1. - n * mu)

        # Neutral landscape is the default
        edges = compute_edges(N, num_types=n, incentive_func=replicator, mu=mu)

        stationary_1 = neutral_stationary(N, alpha, n)
        stationary_2 = approximate_stationary_distribution(edges, convergence_lim=lim)
        for key in stationary_1.keys():
            assert_almost_equal(stationary_1[key], stationary_2[key], places=4)

        # Check that the stationary distribution satistifies balance conditions
        check_detailed_balance(edges, stationary_1)
        check_global_balance(edges, stationary_2)
        check_eigenvalue(edges, stationary_1)

        # Test log versions
        stationary_1 = log_neutral_stationary(N, alpha, n)
        stationary_2 = log_approximate_stationary_distribution(edges, convergence_lim=lim)
        for key in stationary_1.keys():
            assert_almost_equal(stationary_1[key], stationary_2[key], places=4)
