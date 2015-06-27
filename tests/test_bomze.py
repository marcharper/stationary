
from nose.tools import assert_almost_equal, assert_equal, assert_raises, assert_true, assert_less_equal, assert_greater_equal, assert_greater

from stationary.utils.bomze import bomze_matrices

def test_bomze_matrices():
    """
    Check that the data file with the Bomze classification matrices is present
    and loads the correct number of matrices.
    """

    matrices = list(bomze_matrices())
    assert_equal(len(matrices), 49)
