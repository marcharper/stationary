"""
Necessary mathematical functions for the incentive process.
"""

from __future__ import absolute_import

import numpy
from ..utils.math_helpers import multiply_vectors, dot_product


# Fitness Landscapes

def constant_fitness(c):
    """
    Returns a function that is constant on the population state space.
    """

    def f(pop):
        return numpy.array(pop) * numpy.array(c)
    return f


def linear_fitness_landscape(m, self_interaction=True, normalize=False):
    """
    Computes a fitness landscape from a game matrix given by m and a population
    vector (i,j) summing to N.

    Parameters
    ----------
    m: matrix or list of lists
        The game matrix defining the landscape
    self_interaction: bool, True
        Whether players can self-interact, which affects the fitness landscape
    normalize: bool, False
        Whether to normalize the population states (typically not necessary)

    Returns
    -------
    A function on population states (the fitness landscape)
    """

    # m = array of rows
    def f(pop):
        N = sum(pop)
        div = N
        if not self_interaction:
            div = N - 1
        # Normalize population vector
        pop = [x / float(div) for x in pop]
        fitness = []
        for i in range(len(pop)):
            # - m[i][i] if individuals do not interact with themselves.
            f = dot_product(m[i], pop)
            if not self_interaction:
                f -= m[i][i]
            if normalize:
                f = f / float(div)
            fitness.append(f)
        return fitness
    return f


def rock_paper_scissors(a=1, b=1):
    """
    The game matrix for the rock-paper-scissors game.
    """

    return [[0,- b, a], [a, 0, -b], [-b, a, 0]]

# Some people call it rock-scissors-paper
rock_scissors_paper = rock_paper_scissors


# Incentive Functions

def replicator(fitness, q=1, **kwargs):
    """
    The replicator incentive for a power q. For q=1 this reproduces the Moran
    process ratio.

    Parameters
    ----------
    fitness: function
        A fitness landscape
    q: float
        Exponent for the population state

    Returns
    -------
    a function corresponding to the incentive
    """

    if q == 1:
        def f(x):
            return multiply_vectors(x, fitness(x))
        return f

    def g(x):
        y = numpy.power(x, q)
        return y * fitness(x)
    return g


def logit(fitness, beta=1., q=0.):
    """
    The logit incentive for a power q. For q=0 this reproduces the Logit
    process ratio.

    Parameters
    ----------
    fitness: function
        A fitness landscape
    q: float
        Exponent for the population state
    beta: float
        An inverse temperature / strength of selection parameter

    Returns
    -------
    a function corresponding to the incentive.
    """

    if q == 0:
        def f(x):
            return numpy.exp(numpy.array(fitness(x)) * beta)
        return f

    def g(x):
        y = numpy.power(x, q)
        return multiply_vectors(y, numpy.exp(numpy.array(fitness(x)) * beta))
    return g


def fermi(fitness, beta=1., q=1.):
    """
    The Fermi incentive for a power q. For q=0 this reproduces the Fermi process
    ratio. Equal to the logit incentive with q=1.

    Parameters
    ----------
    fitness: function
        A fitness landscape
    q: float
        Exponent for the population state
    beta: float
        An inverse temperature / strength of selection parameter

    Returns
    -------
    a function corresponding to the incentive.
    """

    return logit(fitness, beta=beta, q=1)


def logit2(fitness, beta=1., **kwargs):
    """
    The logit incentive for use with large beta, which approximates the
    best-reply incentive. Uses a log-space calculation.

    Parameters
    ----------
    fitness: function
        A fitness landscape
    beta: float
        An inverse temperature / strength of selection parameter

    Returns
    -------
    a function corresponding to the incentive.
    """

    def g(x):
        i1, i2 = x
        f = fitness(x)
        diff = f[1] - f[0]
        denom = i1+i2*numpy.exp(beta * diff)
        return [i1/denom, i2 * numpy.exp(beta*diff) / denom]
    return g


def simple_best_reply(fitness):
    def g(x):
        f = fitness(x)
        if f[0] > f[1]:
            return [1, 0]
        return [0, 1]
    return g
