import math
import numpy
from math_helpers import multiply_vectors, normalize, dot_product

########################
## Fitness Landscapes ##
########################

def constant_fitness(c):
    def f(pop):
        return numpy.array(pop) * numpy.array(c)
    return f

def linear_fitness_landscape(m, beta=None, self_interaction=True, norm=False):
    """Computes a fitness landscape from a game matrix given by m and a population vector (i,j) summing to N."""
    # m = array of rows
    def f(pop):
        N = sum(pop)
        div = N
        if self_interaction:
            div = N-1
        # Typically unnecessary
        if norm:
            pop = [x / float(div) for x in pop]
        fitness = []
        for i in range(len(pop)):
            # - m[i][i] if individuals do not interact with themselves.
            f = dot_product(m[i], pop)
            if not self_interaction:
                f -= m[i][i]
            fitness.append(f / float(div))
        return fitness
    return f

def rock_scissors_paper(a=1, b=1):
    return [[0,-b,a], [a, 0, -b], [-b, a, 0]]

#########################
## Incentive Functions ##
#########################

## Best Reply is approximable by logit with large Beta

def replicator(fitness, q=1, **kwargs):
    if q == 1:
        def f(x):
            return multiply_vectors(x, fitness(x))
        return f
    def g(x):
        y = []
        for i in range(len(x)):
            y.append(math.pow(x[i], q))
        y = numpy.array(y)
        return y * fitness(x)
    return g

def logit(fitness, beta=1., q=0.):
    if q == 0:
        def f(x):
            return numpy.exp(numpy.array(fitness(x)) * beta)
        return f
    def g(x):
        y = []
        for i in range(len(x)):
            y.append(math.pow(x[i], q))
        y = numpy.array(y)
        return multiply_vectors(y, numpy.exp(numpy.array(fitness(x)) * beta))
    return g

def fermi(fitness, beta=1., q=1.):
    return logit(fitness, beta=beta, q=1)

# Better for large beta for 3x3 games
def logit2(fitness, beta=1.):
    def g(x):
        y = []
        i1, i2 = x
        f = fitness(x)
        diff = f[1] - f[0]
        denom = i1+i2*numpy.exp(beta*diff)
        return [i1/denom, i2 * numpy.exp(beta*diff) / denom]
    return g

# Better for large beta for 3x3 games
def simple_best_reply(fitness):
    def g(x):
        y = []
        i1, i2 = x
        f = fitness(x)
        if f[0] > f[1]:
            return [1,0]
        return [0,1]
    return g
