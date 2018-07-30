import math

import matplotlib
from matplotlib import pyplot
import matplotlib.gridspec as gridspec
from numpy import arange

from stationary import stationary_distribution, convenience
from stationary.utils.math_helpers import normalize

# Font config for plots
font = {'size': 20}
matplotlib.rc('font', **font)


def fixation_probabilities(N, r):
    """
    The fixation probabilities of the classical Moran process.

    Parameters
    ----------
    N: int
        The population size
    r: float
        Relative fitness.
    """

    def phi(N, r, i=1.):
        return (1. - math.pow(r, -i)) / (1. - math.pow(r, -N))

    if r == 0:
        return (0., 1.)
    if r == 1:
        return (1. / N, 1. / N)
    return (phi(N, r), phi(N, 1. / r))


def fixation_comparison(N=20, r=1.2, mu=1e-24):
    """
    Plot the fixation probabilities and the stationary limit.
    """

    fs = []
    ss = []
    diffs = []
    domain = list(arange(0.5, 1.5, 0.01))
    for r in domain:
        game_matrix = [[1, 1], [r, r]]
        edges, s, er = convenience.moran(N, game_matrix, mu, exact=True,
                                         logspace=True)
        fix_1, fix_2 = fixation_probabilities(N, r)
        s_1, s_2 = s[(0, N)], s[(N, 0)]
        f = normalize([fix_1, fix_2])
        fs.append(f[0])
        ss.append(s_1)
        diffs.append(fs[-1] - ss[-1])

    gs = gridspec.GridSpec(2, 1)

    ax1 = pyplot.subplot(gs[0, 0])
    ax2 = pyplot.subplot(gs[1, 0])

    ax1.plot(domain, fs)
    ax1.plot(domain, ss)
    ax1.set_xlabel("Relative fitness $r$")
    ax1.set_ylabel("Fixation Probability $\\rho_A / (\\rho_A + \\rho_B)$")
    ax1.set_title("Fixation Probabilities and Stationary Distribution")

    ax2.plot(domain, diffs)
    ax2.set_xlabel("Relative fitness $r$")
    ax2.set_ylabel("$\\rho_A / (\\rho_A + \\rho_B) - s_{(O, N)}$")

    pyplot.show()


if __name__ == '__main__':
    fixation_comparison(16)
