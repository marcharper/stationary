import math
import numpy
from numpy import log

from incentives import linear_fitness_landscape, replicator
from stationary import approximate_stationary_distribution

from math_helpers import kl_divergence, simplex_generator, one_step_indicies_generator, dot_product, normalize, q_divergence

from matplotlib import pyplot
import ternary

##################################
## Non-constant Population Size ##
##################################

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
def generalized_moran_simulation_transitions(N, fitness_landscape, death_probabilities=None, incentive=None, mu=0.001):
    """Returns a graph of the Markov process corresponding to a generalized Moran process, allowing for uncoupled birth and death processes."""
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
                q = (1. - p) * (birth_q[0] * (1- mu) + birth_q[1] * mu)
                if q > 0:
                    edges.append(((a, b), (a + 1, b), q))
            if b < N:
                q = (1. - p) * (birth_q[0] * mu + birth_q[1] * (1.-mu))
                if q > 0:
                    edges.append(((a, b), (a, b + 1), q))
    return edges

def kl(N, edges, q=1, take_log=False):
    """Computes the KL-div of the expected state with the state, for all states."""
    dist = q_divergence(q)
    e = dict()
    for x, y, w in edges:
        try:
            e[x] += numpy.array(y) * w
        except KeyError:
            e[x]  = numpy.array(y) * w
    d = dict()
    for (i, j), v in e.items():
        # KL doesn't play well on the boundary.
        if i*j == 0:
            continue
        print i,j, v
        r = dist(normalize(v), normalize([i,j]))
        d[(i,j)] = math.sqrt(r) 
    return d

if __name__ == '__main__':
    N = 40
    m = [[1,2],[2,1]]
    fitness_landscape = linear_fitness_landscape(m, self_interaction=True)
    incentive = replicator(fitness_landscape)
    #death_probabilities = sigmoid_death(N)
    #death_probabilities = linear_death(N)
    #death_probabilities = moran_cascade(N, k=1.05)

    death_probabilities = even_death(N)

    edges = generalized_moran_simulation_transitions(N, fitness_landscape, death_probabilities, incentive, mu=0.01)
    s = approximate_stationary_distribution(N, edges, iterations=1000)
    vs = [(v,k) for (k,v) in s.items()]
    vs.sort(reverse=True)
    print vs[:10]

    # Remove boundary states for nice plotting
    for k, v in s.items():
        if k[0] == 0 or k[1] == 0:
            del s[k]

    pyplot.figure()
    ternary.heatmap(s, N)

    pyplot.figure()    
    d = kl(N, edges, q=0)
    vs = [(v,k) for (k,v) in d.items()]
    vs.sort()
    print vs[:10]
    ternary.heatmap(d, N)    
    pyplot.show()
    exit() 
    mins = []
    domain = range(2, N+1)
    
    pyplot.show()
        

        

    