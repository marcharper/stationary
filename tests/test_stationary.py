from __future__ import absolute_import

import numpy

from nose.tools import assert_almost_equal, assert_equal, assert_raises, \
                       assert_true, assert_less_equal, assert_greater_equal

from stationary.stationary import approximate_stationary_distribution, log_approximate_stationary_distribution, neutral_stationary, log_neutral_stationary, edges_to_edge_dict, edges_to_matrix, states_from_edges, entropy_rate, approximate_stationary_distribution_func

from stationary.processes import incentive_process, wright_fisher
from stationary.processes.incentives import replicator, linear_fitness_landscape
from stationary.utils.matrix_checks import check_detailed_balance, check_global_balance, check_eigenvalue

from stationary.utils.math_helpers import kl_divergence_dict

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

def test_incentive_process(lim=1e-14):
    """
    Compare stationary distribution computations to known analytic form for
    neutral landscape for the Moran process.
    """
    for n, N in [(2, 10), (2, 100), (3, 10), (3, 20), (4, 10)]:
        mu = (n-1.)/n * 1./(N+1)
        alpha = N * mu / (n - 1. - n * mu)

        # Neutral landscape is the default
        edges = incentive_process.compute_edges(N, num_types=n,
                                                incentive_func=replicator, mu=mu)

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

        # Test Entropy Rate bounds
        er = entropy_rate(edges, stationary_1)
        h = (2. * n - 1) / n * numpy.log(n)
        assert_less_equal(er, h)
        assert_greater_equal(er, 0)

def test_wright_fisher_2(N=100, lim=1e-12, n=2):
    """Test 2 dimensional Wright-Fisher process."""
    mu = (n - 1.)/n * 1./(N+1)
    m = numpy.ones((n, n)) # neutral landscape
    fitness_landscape = linear_fitness_landscape(m)
    incentive = replicator(fitness_landscape)

    # Wright-Fisher
    wf_edges = wright_fisher.multivariate_transitions(N, incentive, mu=mu, num_types=n)
    stationary = approximate_stationary_distribution(wf_edges, convergence_lim=lim)

    # Check that the stationary distribution satistifies balance conditions
    check_detailed_balance(wf_edges, stationary, places=3)
    check_global_balance(wf_edges, stationary, places=5)
    check_eigenvalue(wf_edges, stationary, places=3)


def test_wright_fisher_3(N=50, lim=1e-16, n=3):
    """Test 3 dimensional Wright-Fisher process."""
    mu = (n - 1.)/n * 1./(N+1)
    m = numpy.ones((n, n)) # neutral landscape
    fitness_landscape = linear_fitness_landscape(m)
    incentive = replicator(fitness_landscape)

    # Wright-Fisher
    edge_func = wright_fisher.multivariate_transitions(N, incentive, mu=mu, num_types=n)
    stationary = approximate_stationary_distribution_func(N, edge_func, convergence_lim=lim)

    # Check that the stationary distribution satistifies balance conditions
    check_detailed_balance(wf_edges, stationary, places=3)
    check_global_balance(edges, stationary, places=4)
    check_eigenvalue(edges, stationary, places=2)
