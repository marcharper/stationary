
from scipy.misc import comb

from nose.tools import assert_almost_equal, assert_equal, assert_raises, assert_true, assert_less_equal, assert_greater_equal, assert_greater

from stationary.utils.math_helpers import simplex_generator

def test_stationary_generator():
    d = 1
    N = 1
    states = set(simplex_generator(N, d))
    expected = set([(0, 1), (1, 0)])
    assert_equal(states, expected)

    N = 2
    states = set(simplex_generator(N, d))
    expected = set([(0, 2), (1, 1), (2, 0)])
    assert_equal(states, expected)

    N = 3
    states = set(simplex_generator(N, d))
    expected = set([(0, 3), (1, 2), (2, 1), (3, 0)])
    assert_equal(states, expected)

    d = 2
    N = 1
    states = set(simplex_generator(N, d))
    expected = set([(1, 0, 0), (0, 1, 0), (0, 0, 1)])
    assert_equal(states, expected)

    N = 2
    states = set(simplex_generator(N, d))
    expected = set([(1, 1, 0), (0, 1, 1), (1, 0, 1), (0, 2, 0), (0, 0, 2), 
                    (2, 0, 0)])
    assert_equal(states, expected)

    for d in range(1, 5):
        for N in range(1, 20):
            states = set(simplex_generator(N, d))
            size = comb(N + d, d, exact=True)
            assert_equal(len(states), size)
