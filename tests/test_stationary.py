from __future__ import absolute_import

import numpy

from nose.tools import assert_almost_equal, assert_equal, assert_raises, assert_true, assert_less_equal, assert_greater_equal, assert_greater

from stationary.stationary import approximate_stationary_distribution, log_approximate_stationary_distribution, neutral_stationary, log_neutral_stationary,  entropy_rate, approximate_stationary_distribution_func

from stationary.processes import incentive_process, wright_fisher
from stationary.processes.incentives import replicator, linear_fitness_landscape
from stationary.utils.matrix_checks import check_detailed_balance, check_global_balance, check_eigenvalue

from stationary.utils.math_helpers import kl_divergence_dict, simplex_generator
from stationary.utils.edges import edges_to_edge_dict, edges_to_matrix, states_from_edges, edge_func_to_edges
from stationary.utils.extrema import find_local_extrema


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

    # Check that the stationary distribution satisfies balance conditions
    for s in [exact_stationary, approx_stationary]:
        check_detailed_balance(edges, s)
        check_global_balance(edges, s)
        check_eigenvalue(edges, s)

def test_stationary_2():
    """
    Test the stationary distribution computations a simple Markov process.
    """

    edges = [(0, 0, 1./3), (0, 1, 1./3), (0, 2, 1./3),
             (1, 0, 1./4), (1, 1, 1./2), (1, 2, 1./4),
             (2, 0, 1./6), (2, 1, 1./3), (2, 2, 1./2),]
    exact_stationary = {0: 6./25, 1: 10./25, 2:9./25}

    approx_stationary = approximate_stationary_distribution(edges)
    for i in [0, 1, 2]:
        assert_almost_equal(exact_stationary[i], approx_stationary[i])

    # Check that the stationary distribution satisfies balance conditions
    for s in [exact_stationary, approx_stationary]:
        check_global_balance(edges, s)
        check_eigenvalue(edges, s)

def test_stationary_3():
    """
    Test the stationary distribution computations a simple Markov process.
    """

    edges = [(0, 0, 0), (0, 1, 1), (0, 2, 0), (0, 3, 0),
             (1, 0, 1./3), (1, 1, 0), (1, 2, 2./3), (1, 3, 0),
             (2, 0, 0), (2, 1, 2./3), (2, 2, 0), (2, 3, 1./3),
             (3, 0, 0), (3, 1, 0), (3, 2, 1), (3, 3, 0)]
    exact_stationary = {0: 1./8, 1: 3./8, 2: 3./8, 3: 1./8}

    approx_stationary = approximate_stationary_distribution(edges)
    for i in [0, 1, 2, 3]:
        assert_almost_equal(exact_stationary[i], approx_stationary[i])

    # Check that the stationary distribution satisfies balance conditions
    for s in [exact_stationary, approx_stationary]:
        check_detailed_balance(edges, s)
        check_global_balance(edges, s)
        check_eigenvalue(edges, s)

def test_stationary_4():
    """
    Test the stationary distribution computations a simple Markov process.
    """

    edges = [(0, 0, 1./2), (0, 1, 1./2), (0, 2, 0), (0, 3, 0),
             (1, 0, 1./6), (1, 1, 1./2), (1, 2, 1./3), (1, 3, 0),
             (2, 0, 0), (2, 1, 1./3), (2, 2, 1./2), (2, 3, 1./6),
             (3, 0, 0), (3, 1, 0), (3, 2, 1./2), (3, 3, 1./2)]
    exact_stationary = {0: 1./8, 1: 3./8, 2: 3./8, 3: 1./8}

    approx_stationary = approximate_stationary_distribution(edges)
    for i in [0, 1, 2, 3]:
        assert_almost_equal(exact_stationary[i], approx_stationary[i])

    # Check that the stationary distribution satisfies balance conditions
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

def test_incentive_process_k(lim=1e-14):
    """
    Compare stationary distribution computations to known analytic form for
    neutral landscape for the Moran process.
    """
    for k in [1, 2, 10,]:
        #(2, 10), (2, 100), 
        for n, N in [(3, 10), (3, 20)]:
            mu = (n-1.)/n * 1./(N+1)
            m = numpy.ones((n, n)) # neutral landscape
            fitness_landscape = linear_fitness_landscape(m)
            incentive = replicator(fitness_landscape)

            # Neutral landscape is the default
            edges = incentive_process.k_fold_incentive_transitions(N, incentive, num_types=n, mu=mu, k=k)

            stationary_1 = approximate_stationary_distribution(edges, convergence_lim=lim)

            # Check that the stationary distribution satistifies balance conditions
            check_detailed_balance(edges, stationary_1)
            check_global_balance(edges, stationary_1)
            check_eigenvalue(edges, stationary_1)

def test_extrema_moran(lim=1e-10):
    """
    For small mu, the Moran process is maximal on the corner points of the
    simplex and minimal in the center. Test that this happens.
    """

    for n, N, mins, maxes in [(2, 100, [(50, 50)], [(0, 100), (100,0)]),
                              (3, 45, [(15, 15, 15)], [(45,0,0), (0, 45, 0), (0, 0, 45)])]:
        mu = 1./N**3
        edges = incentive_process.compute_edges(N, num_types=n,
                                                incentive_func=replicator, mu=mu)
        s = approximate_stationary_distribution(edges, convergence_lim=lim)
        assert_equal(find_local_extrema(s, extremum="min"), set(mins))
        assert_equal(find_local_extrema(s, extremum="max"), set(maxes))
        extrema = set(mins).union(set(maxes))
        s2 = incentive_process.kl(edges, q_d=0)
        s3 = [(v, k) for (k, v) in s2.items()]
        s3.sort()
        print s3[0:10]
        print find_local_extrema(s2, extremum="min")
        assert_equal(find_local_extrema(s2, extremum="min"), extrema)

def test_extrema_wf(lim=1e-10):
    """
    For small mu, the Wright-Fisher process is minimal in the center.
    Test that this happens.
    """

    for n, N, mins in [(2, 60, [(30,30)]), (3, 45, [(15, 15, 15)])]:
        mu = 1./N**3
        m = numpy.ones((n, n)) # neutral landscape
        fitness_landscape = linear_fitness_landscape(m)
        incentive = replicator(fitness_landscape)

        if n == 2:
            wf_edges = wright_fisher.multivariate_transitions(N, incentive, mu=mu, num_types=n)
            s = approximate_stationary_distribution(wf_edges, convergence_lim=lim)
            #s2 = wright_fisher.kl(

        if n == 3:
        # Wright-Fisher
            edge_func = wright_fisher.multivariate_transitions(N, incentive, mu=mu, num_types=n)
            s = approximate_stationary_distribution_func(N, edge_func, convergence_lim=lim)

        assert_equal(find_local_extrema(s, extremum="min"), set(mins))


def test_wright_fisher_2(N=100, lim=1e-12, n=2):
    """Test 2 dimensional Wright-Fisher process."""
    mu = (n - 1.)/n * 1./(N+1)
    m = numpy.ones((n, n)) # neutral landscape
    fitness_landscape = linear_fitness_landscape(m)
    incentive = replicator(fitness_landscape)

    # Wright-Fisher
    wf_edges = wright_fisher.multivariate_transitions(N, incentive, mu=mu, num_types=n)
    s = approximate_stationary_distribution(wf_edges, convergence_lim=lim)

    # Check that the stationary distribution satistifies balance conditions
    check_detailed_balance(wf_edges, s, places=3)
    check_global_balance(wf_edges, s, places=5)
    check_eigenvalue(wf_edges, s, places=3)


def test_wright_fisher_3(N=30, lim=1e-16, n=3):
    """Test 3 dimensional Wright-Fisher process. Only use with small N."""
    mu = (n - 1.)/n * 1./(N+1)
    m = numpy.ones((n, n)) # neutral landscape
    fitness_landscape = linear_fitness_landscape(m)
    incentive = replicator(fitness_landscape)

    # Wright-Fisher
    edge_func = wright_fisher.multivariate_transitions(N, incentive, mu=mu, num_types=n)
    s = approximate_stationary_distribution_func(N, edge_func, convergence_lim=lim)
    wf_edges = edge_func_to_edges(edge_func, N)

    # Check that the stationary distribution satistifies balance conditions
    check_detailed_balance(wf_edges, s, places=3)
    check_global_balance(wf_edges, s, places=4)
    check_eigenvalue(wf_edges, s, places=2)
