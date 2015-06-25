from __future__ import absolute_import

from nose.tools import assert_almost_equal, assert_equal, assert_raises, \
                       assert_true

from stationary.stationary import approximate_stationary_distribution, log_approximate_stationary_distribution

def test_stationary(t1=0.4, t2=0.6):
    edges = [(0, 1, t1), (0, 0, 1. - t1), (1, 0, t2), (1, 1, 1. - t2)]
    s_0 = 1./(1. + t1 / t2)
    exact_stationary = {0: s_0, 1: 1 - s_0}

    for func in [approximate_stationary_distribution, log_approximate_stationary_distribution]:
        approx_stationary = func(edges)
        print exact_stationary, approx_stationary
        for i in [0, 1]:
            assert_almost_equal(exact_stationary[i], approx_stationary[i])

