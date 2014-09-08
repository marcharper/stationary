from collections import defaultdict

from mpsim.stationary import Cache, Graph, stationary_distribution_generator
from math_helpers import kl_divergence, kl_divergence_dict, simplex_generator

import numpy
from numpy import array, log, exp

from scipy.special import gammaln
from scipy.misc import logsumexp

numpy.seterr(all="print")

# Reference: http://www.statslab.cam.ac.uk/~frank/BOOKS/book/ch1.pdf

#############
## Helpers ##
#############

def edges_to_matrix(edges):
    # Enumerate states so we can put them in a matrix.
    all_states = set()
    for (a,b,v) in edges:
        all_states.add(a)
        all_states.add(b)
    enumeration = dict(zip(all_states, range(len(all_states))))
    # Build a matrix for the transitions
    mat = numpy.zeros((len(all_states), len(all_states)))
    for (a, b, v) in edges:
        mat[enumeration[a]][enumeration[b]] = v
    return mat, all_states, enumeration

def edges_to_edge_dict(edges):
    edge_dict = dict()
    for e1, e2, v in edges:
        edge_dict[(e1,e2)] = v
    return edge_dict

##############################
## Stationary Distributions ##
##############################

### Exact computations for reversible processes. Use at your own risk! No check for reversibility is performed

def exact_stationary_distribution(edges, num_players=None, initial_index=1, initial=None, N=None):
    """Computes the stationary distribution of a reversible process exactly. 'edges' is actually an 'edge_dict'"""
    if not N:
        N = sum(edges.keys()[0][0])
    if not num_players:
        num_players = len(edges.keys()[0][0])
    if not initial:
        initial = [N//num_players]*(num_players)
        initial[-1] = N - (num_players-1) * (N//num_players)
        #initial[initial_index] = N  # (N,0,..,0)
    initial = tuple(initial)
    d = dict()
    for state in simplex_generator(N, num_players-1):
        # Take a path from initial to state.
        seq = [initial]
        e = list(seq[-1])
        for i in range(0, num_players):
            while e[i] < state[i]:
                for j in range(0, num_players):
                    if e[j] > state[j]:
                        break
                e[j] = e[j] - 1
                e[i] = e[i] + 1
                seq.append(tuple(e))
            while e[i] > state[i]:
                for j in range(0, num_players):
                    if e[j] < state[j]:
                        break
                e[j] = e[j] + 1
                e[i] = e[i] - 1
                seq.append(tuple(e))
        s = 1.
        for index in range(len(seq)-1):
            e, f = seq[index], seq[index+1]
            s *= edges[(e,f)] / edges[(f, e)]
        d[state] = s
    s0 = 1./(sum([v for v in d.values()]))
    for key, v in d.items():
        d[key] = s0 * v
    return d

def log_exact_stationary_distribution(edges, num_players=None, initial=None, no_boundary=False, N=None):
    """Same as the exact calculation but assumes edges have been log-ed for greater precision."""
    if not N:
        N = sum(edges.keys()[0][0])
    if not num_players:
        num_players = len(edges.keys()[0][0])
    if not initial:
        initial = [N//num_players]*(num_players)
        initial[-1] = N - (num_players-1) * (N//num_players)
    initial = tuple(initial)
    d = dict()
    for state in simplex_generator(N, num_players-1):
        if no_boundary:
            is_boundary = False
            for i in state:
                if i == 0:
                    is_boundary = True
                    break
            if is_boundary:
                continue
        # Take a path from initial to state.
        seq = [initial]
        e = list(seq[-1])
        for i in range(0, num_players):
            while e[i] < state[i]:
                for j in range(0, num_players):
                    if e[j] > state[j]:
                        break
                e[j] = e[j] - 1
                e[i] = e[i] + 1
                seq.append(tuple(e))
            while e[i] > state[i]:
                for j in range(0, num_players):
                    if e[j] < state[j]:
                        break
                e[j] = e[j] + 1
                e[i] = e[i] - 1
                seq.append(tuple(e))
        s = 0.
        for index in range(len(seq)-1):
            e, f = seq[index], seq[index+1]
            s += edges[(e,f)] - edges[(f, e)]
        d[state] = s
    s0 = logsumexp([v for v in d.values()])
    for key, v in d.items():
        d[key] = exp(v-s0)
    return d

### Approximate stationary distributions computed by by sparse matrix multiplications. Produces correct results and uses little memory but is likely not the most CPU efficient implementation in general (e.g. and eigenvector calculator may be better).

### For the Wright-Fisher process use the direct implementation stationary function in wright_fisher.py (that doesn't rely on the stationary generator function from mpsim)

def approximate_stationary_distribution(edges, iterations=None, convergence_lim=1e-8):
    """Essentially raising the transition probabilities matrix to a large power using a sparse-matrix implementation."""
    g = Graph()
    g.add_edges(edges)
    #g.normalize_weights()
    cache = Cache(g)
    gen = stationary_distribution_generator(cache)
    previous_ranks = None
    for i, ranks in enumerate(gen):
        if i > 200:
            if i % 10:
                s = kl_divergence(ranks, previous_ranks)
                if s < convergence_lim:
                    break
        if iterations:
            if i == iterations:
                break
        previous_ranks = ranks
    print i
    # Reverse enumeration
    d = dict()
    for m, r in enumerate(ranks):
        state = cache.inv_enum[m]
        d[(state)] = r
    return d

def log_approximate_stationary_distribution(edges, iterations=None, convergence_lim=1e-12):
    """Essentially raising the transition probabilities matrix to a large power using a sparse-matrix implementation."""
    g = Graph()
    g.add_edges(edges)
    #g.normalize_weights()
    cache = Cache(g)
    gen = log_stationary_distribution_generator(cache)
    previous_ranks = None
    for i, ranks in enumerate(gen):
        if i > 100:
            if i % 10:
                s = kl_divergence(ranks, previous_ranks)
                if s < convergence_lim:
                    break
        if iterations:
            if i == iterations:
                break
        previous_ranks = ranks
    # Reverse enumeration
    d = dict()
    for m, r in enumerate(ranks):
        state = cache.inv_enum[m]
        d[(state)] = r
    return d

def compute_stationary(edges, exact=False, convergence_lim=1e-13):
    """Convenience Function for computing stationary distribution."""
    if not exact:
        # Approximate Calculation
        s = approximate_stationary_distribution(edges, convergence_lim=convergence_lim)
    else:
        # Exact Calculuation
        edge_dict = edges_to_edge_dict(edges)
        s = exact_stationary_distribution(edge_dict)
    return s


###################################
## Neutral landscape / Dirichlet ##
###################################

### Exact stationary distribution for the neutral landscape for any n, do not use for any other landscape! ###

## Helpers ##

def inc_factorial(x,n):
    p = 1.
    for i in range(0, n):
       p *= (x + i)
    return p

def factorial(i):
    p = 1.
    for j in range(2, i+1):
        p *= j
    return p

def log_inc_factorial(x,n):
    p = 1.
    for i in range(0, n):
       p += log(x + i)
    return p

def log_factorial(i):
    p = 1.
    for j in range(2, i+1):
        p += log(j)
    return p

### Stationary calculators. For n > ~150, need to use log-space implementation to prevent under/over-flow ##

def neutral_stationary(N, alpha, n=3):
    """Computes the stationary distribution of the neutral landscape."""
    if N > 100:
        return log_neutral_stationary(N, alpha, n=n)
    d2 = dict()
    for state in simplex_generator(N, n-1):
        t = 1.
        for i in state:
            t *= inc_factorial(alpha, i) / factorial(i)
        t *= factorial(N) / inc_factorial(n * alpha, N)        
        d2[state] = t
    return d2

def log_neutral_stationary(N, alpha, n=3):
    """Computes the stationary distribution of the neutral landscape in log space."""
    d2 = dict()
    for state in simplex_generator(N, n-1):
        #print state
        t = 0.
        for i in state:
            t += log_inc_factorial(alpha, i) - log_factorial(i)
        t += log_factorial(N) - log_inc_factorial(n * alpha, N)        
        d2[state] = exp(t)
    return d2

################
# Entropy Rate #
################

def entropy_rate(edges, stationary):
    """Computes the entropy rate given the edges of the process and the stationary distribution."""
    e = defaultdict(float)
    for a,b,v in edges:
        e[a] -= stationary[a] * v * log(v)
    return sum(e.values())

def entropy_rate_func(N, edge_func, s):
    """Computes entropy rate for a process with a large transition matrix, defined by a transition function (edge_func) rather than a list of weighted edges."""
    e = defaultdict(float)
    for a in simplex_generator(N):
        for b in simplex_generator(N):
            v = edge_func(a,b)
            e[a] -= s[a] * v * log(v)
    return sum(e.values())
