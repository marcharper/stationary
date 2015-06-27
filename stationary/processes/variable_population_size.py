import numpy

from ..utils.math_helpers import dot_product, normalize, q_divergence


# Random-death probability distributions

def moran_death(N):
    def p(pop):
        s = sum(pop)
        if s == N + 1:
            return 1
        return 0
    return p

def moran_cascade(N, k=2):
    def p(pop):
        s = sum(pop)
        return math.pow(k, s-N)
    return p

def discrete_sigmoid(t, k_1=0.1, k_2=-1.1):
    # Adapted from https://dinodini.wordpress.com/2010/04/05/normalized-tunable-sigmoid-functions/
    k_2 = -1 - k_1
    if t < 0:
        return 0
    if t <= 0.5:
        return k_1 * t / (k_1 - 2*t + 1)
    else:
        return 0.5 + 0.5*k_2 * (2*t - 1.) / (k_2 - (2* t - 1.) + 1)

def sigmoid_death(N, k_1=2, k_2=2):
    def p(pop):
        s = sum(pop)
        return discrete_sigmoid(float(s) / N, k_1=k_1, k_2=k_2)
    return p

def linear_death(N):
    def p(pop):
        s = sum(pop)
        if s == 1:
            return 0
        return float(s) / N
    return p

def even_death(N):
    def p(pop):
        s = sum(pop)
        if s == 1 or s == N:
            return 0
        return 0.5
    return p

# 2d moran-like process separating birth and death processes
def variable_population_transitions(N, fitness_landscape, death_probabilities=None, incentive=None, mu=0.001):
    """
    Computes transition probabilities for the incentive process on two types
    for a population of varying size.

    Parameters
    ----------
    N: int
        Max population size / simplex divisor
    fitness_landscape, function
        The fitness landscape
    death_probabilities, function
        A function returning probalities of a death event
    incentive: function
        An incentive function from incentives.py
    mu: float, 0.001
        The mutation rate of the process
    """

    if not death_probabilities:
        death_probabilities = moran_death(N)
    edges = []
    # Possible states are (a, b) with 0 < a + b <= N where a is the number of A individuals and B is the number of B individuals.
    for a in range(0, N + 1):
        for b in range(0, N + 1 - a):
            # Death events.
            if a + b == 0:
                continue
            #print a, b
            p = death_probabilities((a, b))
            if a > 0 and b > 0:
                q = p * float(a) / (a + b)
                if q > 0:
                    edges.append(((a, b), (a - 1, b), q))
            #if b > 0:
                q = p * float(b) / (a + b)
                if q > 0:
                    edges.append(((a, b), (a, b - 1), q))
            # Birth events.
            if a + b >= N:
                continue
            if incentive:
                birth_q = normalize(incentive([a,b]))
            else:
                birth_q = normalize(multiply_vectors([a, b], fitness_landscape([a,b])))
            if a < N:
                q = (1. - p) * (birth_q[0] * (1 - mu) + birth_q[1] * mu)
                if q > 0:
                    edges.append(((a, b), (a + 1, b), q))
            if b < N:
                q = (1. - p) * (birth_q[0] * mu + birth_q[1] * (1.-mu))
                if q > 0:
                    edges.append(((a, b), (a, b + 1), q))
    return edges
