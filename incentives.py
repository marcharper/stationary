import math
import numpy
from math_helpers import multiply_vectors, normalize, dot_product

def constant_fitness(c):
    def f(pop):
        return numpy.array(pop) * numpy.array(c)
    return f

def linear_fitness_landscape(m, beta=None, self_interaction=True):
    """Computes a fitness landscape from a game matrix given by m and a population vector (i,j) summing to N."""
    # m = array of rows
    def f(pop):
        N = sum(pop)
        if self_interaction:
            div = N
        else:
            div = N-1
        #pop = [x / float(div) for x in pop]
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

def replicator(fitness, q=1):
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

#def log_logit(fitness, beta=1., q=0.):
    #def g(x):
        #y = []
        #for i in range(len(x)):
            ## Fix this
            #if (q == 0):
                #y.append(0)
            #elif (x[i] == 0):
                #y.append(float('-inf'))
            #else:
                #y.append(q* math.log(x[i]))
        #y = numpy.array(y)
        #return y + numpy.array(fitness(x)) * beta
    #return g

def fermi(fitness, beta=1., q=1.):
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

#def best_reply(fitness):
    #"""Compute best reply to fitness landscape at state."""
    #def g(state):
        #f = fitness(state)
        #try:
            #dim = state.size
        #except AttributeError:
            #state = numpy.array(state)
            #dim = state.size
        #replies = []
        #for i in range(dim):
            #x = numpy.zeros(dim)
            #x[i] = 1
            #replies.append(numpy.dot(x, f))
        #replies = numpy.array(replies)
        #i = numpy.argmax(replies)
        #x = numpy.zeros(dim)
        #x[i] = 1
        #return x
    #return g
    