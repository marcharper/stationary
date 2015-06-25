from __future__ import absolute_import

from nose.tools import assert_almost_equal, assert_equal, assert_raises, \
                       assert_true

from stationary.stationary import approximate_stationary_distribution, log_approximate_stationary_distribution, neutral_stationary, log_neutral_stationary

from processes.incentive_process import compute_edges
from processes.incentives import replicator

def test_stationary(t1=0.4, t2=0.6):
    edges = [(0, 1, t1), (0, 0, 1. - t1), (1, 0, t2), (1, 1, 1. - t2)]
    s_0 = 1./(1. + t1 / t2)
    exact_stationary = {0: s_0, 1: 1 - s_0}

    for func in [approximate_stationary_distribution, log_approximate_stationary_distribution]:
        approx_stationary = func(edges)
        for i in [0, 1]:
            assert_almost_equal(exact_stationary[i], approx_stationary[i])

def test_neutral(N=100.):
    for n, N in [(2, 10), (2, 100), (3, 10), (3, 20), (4, 10)]:
        mu = 1. / N
        # Neutral landscape is the default
        edges = compute_edges(N, num_types=n, incentive_func=replicator, mu=mu)
        alpha = N * mu / (n - 1. - n * mu)

        stationary_1 = neutral_stationary(N, alpha, n)
        stationary_2 = approximate_stationary_distribution(edges, convergence_lim=1e-12)
        for key in stationary_1.keys():
            assert_almost_equal(stationary_1[key], stationary_2[key], places=3)
        # Test log versions
        stationary_1 = log_neutral_stationary(N, alpha, n)
        stationary_2 = log_approximate_stationary_distribution(edges, convergence_lim=1e-12)
        for key in stationary_1.keys():
            assert_almost_equal(stationary_1[key], stationary_2[key], places=3)

