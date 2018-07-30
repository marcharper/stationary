from __future__ import absolute_import

import numpy

from nose.tools import (
    assert_almost_equal, assert_equal, assert_raises, assert_true,
    assert_less_equal, assert_greater_equal, assert_greater
)

from stationary import stationary_distribution, entropy_rate
from stationary.processes import incentive_process, wright_fisher
from stationary.processes.incentives import (
    replicator, logit, fermi, linear_fitness_landscape)
from stationary.utils.matrix_checks import (
    check_detailed_balance, check_global_balance, check_eigenvalue)

from stationary.utils import expected_divergence
from stationary.utils.math_helpers import simplex_generator
from stationary.utils.edges import (
    states_from_edges, edge_func_to_edges, power_transitions)
from stationary.utils.extrema import (
    find_local_minima, find_local_maxima, inflow_outflow)


# Test Generic processes

def test_stationary(t1=0.4, t2=0.6):
    """
    Test the stationary distribution computations a simple Markov process.
    """

    edges = [(0, 1, t1), (0, 0, 1. - t1), (1, 0, t2), (1, 1, 1. - t2)]
    s_0 = 1./(1. + t1 / t2)
    exact_stationary = {0: s_0, 1: 1 - s_0}

    for logspace in [True, False]:
        s = stationary_distribution(edges, logspace=logspace)
        # Check that the stationary distribution satisfies balance conditions
        check_detailed_balance(edges, s)
        check_global_balance(edges, s)
        check_eigenvalue(edges, s)
        # Check that the approximation converged to the exact distribution
        for key in s.keys():
            assert_almost_equal(exact_stationary[key], s[key])


def test_stationary_2():
    """
    Test the stationary distribution computations a simple Markov process.
    """

    edges = [(0, 0, 1./3), (0, 1, 1./3), (0, 2, 1./3),
             (1, 0, 1./4), (1, 1, 1./2), (1, 2, 1./4),
             (2, 0, 1./6), (2, 1, 1./3), (2, 2, 1./2),]
    exact_stationary = {0: 6./25, 1: 10./25, 2:9./25}

    for logspace in [True, False]:
        s = stationary_distribution(edges, logspace=logspace)
        # Check that the stationary distribution satisfies balance conditions
        check_global_balance(edges, s)
        check_eigenvalue(edges, s)
        # Check that the approximation converged to the exact distribution
        for key in s.keys():
            assert_almost_equal(exact_stationary[key], s[key])


def test_stationary_3():
    """
    Test the stationary distribution computations a simple Markov process.
    """

    edges = [(0, 0, 0), (0, 1, 1), (0, 2, 0), (0, 3, 0),
             (1, 0, 1./3), (1, 1, 0), (1, 2, 2./3), (1, 3, 0),
             (2, 0, 0), (2, 1, 2./3), (2, 2, 0), (2, 3, 1./3),
             (3, 0, 0), (3, 1, 0), (3, 2, 1), (3, 3, 0)]
    exact_stationary = {0: 1./8, 1: 3./8, 2: 3./8, 3: 1./8}

    for logspace in [True, False]:
        s = stationary_distribution(edges, logspace=logspace)
        # Check that the stationary distribution satisfies balance conditions
        check_detailed_balance(edges, s)
        check_global_balance(edges, s)
        check_eigenvalue(edges, s)
        # Check that the approximation converged to the exact distribution
        for key in s.keys():
            assert_almost_equal(exact_stationary[key], s[key])


def test_stationary_4():
    """
    Test the stationary distribution computations a simple Markov process.
    """

    edges = [(0, 0, 1./2), (0, 1, 1./2), (0, 2, 0), (0, 3, 0),
             (1, 0, 1./6), (1, 1, 1./2), (1, 2, 1./3), (1, 3, 0),
             (2, 0, 0), (2, 1, 1./3), (2, 2, 1./2), (2, 3, 1./6),
             (3, 0, 0), (3, 1, 0), (3, 2, 1./2), (3, 3, 1./2)]
    exact_stationary = {0: 1./8, 1: 3./8, 2: 3./8, 3: 1./8}

    for logspace in [True, False]:
        s = stationary_distribution(edges, logspace=logspace)
        # Check that the stationary distribution satisfies balance conditions
        check_detailed_balance(edges, s)
        check_global_balance(edges, s)
        check_eigenvalue(edges, s)
        # Check that the approximation converged to the exact distribution
        for key in s.keys():
            assert_almost_equal(exact_stationary[key], s[key])

## Test Moran / Incentive Processes


def test_incentive_process(lim=1e-14):
    """
    Compare stationary distribution computations to known analytic form for
    neutral landscape for the Moran process.
    """

    for n, N in [(2, 10), (2, 40), (3, 10), (3, 20), (4, 10)]:
        mu = (n - 1.) / n * 1./ (N + 1)
        alpha = N * mu / (n - 1. - n * mu)

        # Neutral landscape is the default
        edges = incentive_process.compute_edges(N, num_types=n,
                                                incentive_func=replicator, mu=mu)
        for logspace in [False, True]:
            stationary_1 = incentive_process.neutral_stationary(
                N, alpha, n, logspace=logspace)
            for exact in [False, True]:
                stationary_2 = stationary_distribution(
                    edges, lim=lim, logspace=logspace, exact=exact)
                for key in stationary_1.keys():
                    assert_almost_equal(
                        stationary_1[key], stationary_2[key], places=4)

        # Check that the stationary distribution satisfies balance conditions
        check_detailed_balance(edges, stationary_1)
        check_global_balance(edges, stationary_1)
        check_eigenvalue(edges, stationary_1)

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
        for n, N in [(2, 20), (2, 50), (3, 10), (3, 20)]:
            mu = (n-1.)/n * 1./(N+1)
            m = numpy.ones((n, n)) # neutral landscape
            fitness_landscape = linear_fitness_landscape(m)
            incentive = replicator(fitness_landscape)

            # Neutral landscape is the default
            edges = incentive_process.k_fold_incentive_transitions(
                N, incentive, num_types=n, mu=mu, k=k)
            stationary_1 = stationary_distribution(edges, lim=lim)

            # Check that the stationary distribution satisfies balance
            # conditions
            check_detailed_balance(edges, stationary_1)
            check_global_balance(edges, stationary_1)
            check_eigenvalue(edges, stationary_1)

            # Also check edge_func calculation
            edges = incentive_process.multivariate_transitions(
                N, incentive, num_types=n, mu=mu)
            states = states_from_edges(edges)
            edge_func = power_transitions(edges, k)
            stationary_2 = stationary_distribution(
                edge_func, states=states, lim=lim)

            for key in stationary_1.keys():
                assert_almost_equal(
                    stationary_1[key], stationary_2[key], places=5)


def test_extrema_moran(lim=1e-16):
    """
    Test for extrema of the stationary distribution.
    """
    n = 2
    for N, maxes, mins in [(60, [(30, 30)], [(60, 0), (0, 60)]),
                           (100, [(50, 50)], [(100, 0), (0, 100)])]:
        mu = 1. / N
        edges = incentive_process.compute_edges(N, num_types=n,
                                                incentive_func=replicator, mu=mu)

        s = stationary_distribution(edges, lim=lim)
        assert_equal(find_local_maxima(s), set(maxes))
        assert_equal(find_local_minima(s), set(mins))


def test_extrema_moran_2(lim=1e-16):
    """
    Test for extrema of the stationary distribution.
    """
    n = 2
    N = 100
    mu = 1. / 1000
    m = [[1, 2], [3, 1]]
    maxes = set([(33, 67), (100,0), (0, 100)])
    fitness_landscape = linear_fitness_landscape(m)
    incentive = replicator(fitness_landscape)
    edges = incentive_process.multivariate_transitions(N, incentive, num_types=n, mu=mu)
    s = stationary_distribution(edges, lim=lim)
    s2 = expected_divergence(edges, q_d=0)

    assert_equal(find_local_maxima(s), set(maxes))
    assert_equal(find_local_minima(s2), set(maxes))


def test_extrema_moran_3(lim=1e-12):
    """
    Test for extrema of the stationary distribution.
    """
    n = 2
    N = 100
    mu = 6./ 25
    m = [[1, 0], [0, 1]]
    maxes = set([(38, 62), (62, 38)])
    mins = set([(50, 50), (100, 0), (0, 100)])
    fitness_landscape = linear_fitness_landscape(m)
    incentive = replicator(fitness_landscape)
    edges = incentive_process.multivariate_transitions(N, incentive, num_types=n, mu=mu)
    s = stationary_distribution(edges, lim=lim)
    flow = inflow_outflow(edges)

    for q_d in [0, 1]:
        s2 = expected_divergence(edges, q_d=1)
        assert_equal(find_local_maxima(s), set(maxes))
        assert_equal(find_local_minima(s), set(mins))
        assert_equal(find_local_minima(s2), set([(50,50), (40, 60), (60, 40)]))
        assert_equal(find_local_maxima(flow), set(mins))


def test_extrema_moran_4(lim=1e-16):
    """
    Test for extrema of the stationary distribution.
    """
    n = 3
    N = 60
    mu = 3./ (2 * N)
    m = [[0, 1, 1], [1, 0, 1], [1, 1, 0]]
    maxes = set([(20,20,20)])
    mins = set([(0, 0, 60), (0, 60, 0), (60, 0, 0)])
    fitness_landscape = linear_fitness_landscape(m)
    incentive = logit(fitness_landscape, beta=0.1)
    edges = incentive_process.multivariate_transitions(N, incentive, num_types=n, mu=mu)
    s = stationary_distribution(edges, lim=lim)
    s2 = expected_divergence(edges, q_d=0)

    assert_equal(find_local_maxima(s), set(maxes))
    assert_equal(find_local_minima(s), set(mins))
    assert_equal(find_local_minima(s2), set(maxes))
    assert_equal(find_local_maxima(s2), set(mins))


def test_extrema_moran_5(lim=1e-16):
    """
    Test for extrema of the stationary distribution.
    """
    n = 3
    N = 60
    mu = (3./2) * 1./N
    m = [[0, 1, 1], [1, 0, 1], [1, 1, 0]]
    maxes = set([(20, 20, 20), (0, 0, 60), (0, 60, 0), (60, 0, 0),
                 (30, 0, 30), (0, 30, 30), (30, 30, 0)])
    fitness_landscape = linear_fitness_landscape(m)
    incentive = fermi(fitness_landscape, beta=0.1)
    edges = incentive_process.multivariate_transitions(
        N, incentive, num_types=n, mu=mu)

    s = stationary_distribution(edges, lim=lim)
    s2 = expected_divergence(edges, q_d=0)
    flow = inflow_outflow(edges)

    # These sets should all correspond
    assert_equal(find_local_maxima(s), set(maxes))
    assert_equal(find_local_minima(s2), set(maxes))
    assert_equal(find_local_minima(flow), set(maxes))

    # The minima are pathological
    assert_equal(find_local_minima(s),
                 set([(3, 3, 54), (3, 54, 3), (54, 3, 3)]))
    assert_equal(find_local_maxima(s2),
                 set([(4, 52, 4), (4, 4, 52), (52, 4, 4)]))
    assert_equal(find_local_maxima(flow),
                 set([(1, 58, 1), (1, 1, 58), (58, 1, 1)]))


def test_wright_fisher(N=20, lim=1e-10, n=2):
    """Test 2 dimensional Wright-Fisher process."""
    for n in [2, 3]:
        mu = (n - 1.) / n * 1. / (N + 1)
        m = numpy.ones((n, n)) # neutral landscape
        fitness_landscape = linear_fitness_landscape(m)
        incentive = replicator(fitness_landscape)

        # Wright-Fisher
        for low_memory in [True, False]:
            edge_func = wright_fisher.multivariate_transitions(
                N, incentive, mu=mu, num_types=n, low_memory=low_memory)
            states = list(simplex_generator(N, d=n-1))
            for logspace in [False, True]:
                s = stationary_distribution(
                    edge_func, states=states, iterations=200, lim=lim,
                    logspace=logspace)
                wf_edges = edge_func_to_edges(edge_func, states)

                er = entropy_rate(wf_edges, s)
                assert_greater_equal(er, 0)

                # Check that the stationary distribution satistifies balance
                # conditions
                check_detailed_balance(wf_edges, s, places=2)
                check_global_balance(wf_edges, s, places=4)
                check_eigenvalue(wf_edges, s, places=2)


def test_extrema_wf(lim=1e-10):
    """
    For small mu, the Wright-Fisher process is minimal in the center.
    Test that this happens.
    """

    for n, N, mins in [(2, 40, [(20, 20)]), (3, 30, [(10, 10, 10)])]:
        mu = 1. / N ** 3
        m = numpy.ones((n, n)) # neutral landscape
        fitness_landscape = linear_fitness_landscape(m)
        incentive = replicator(fitness_landscape)

        edge_func = wright_fisher.multivariate_transitions(
            N, incentive, mu=mu, num_types=n)
        states = list(simplex_generator(N, d=n-1))
        s = stationary_distribution(
            edge_func, states=states, iterations=4*N, lim=lim)
        assert_equal(find_local_minima(s), set(mins))
        er = entropy_rate(edge_func, s, states=states)
        assert_greater_equal(er, 0)

