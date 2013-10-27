"""Compute E(x), KL, plot heatmap; plot stationary."""
import os

from mpsim.stationary import Cache
from incentives import *
import incentive_process

from math_helpers import kl_divergence, kl_divergence_dict

import numpy
from numpy import array, log, exp

from scipy.special import gammaln
from scipy.misc import logsumexp

numpy.seterr(all="print")

# http://www.statslab.cam.ac.uk/~frank/BOOKS/book/ch1.pdf

#############################
## Stationary Distribution ##
#############################

def exact_stationary_distribution(N, edges, num_players=None, initial_index=1, initial=None):
    """"""
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

def log_exact_stationary_distribution(N, edges, num_players=None, initial=None, no_boundary=False):
    """Assumes edges have been log-ed"""
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

def approximate_stationary_distribution(N, edges, iterations=None, convergence_lim=1e-18):
    """Essentially raising the transition probabilities matrix to a large power using a sparse-matrix implementation."""
    g = Graph()
    g.add_edges(edges)
    #g.normalize_weights()
    cache = Cache(g)
    gen = stationary_distribution_generator(cache)
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
    print i
    # Reverse enumeration
    d = dict()
    for m, r in enumerate(ranks):
        state = cache.inv_enum[m]
        d[(state)] = r
    return d
    exit()

def log_approximate_stationary_distribution(N, edges, iterations=None, convergence_lim=1e-13):
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

def transition_test(N=100, mu=0.01, no_boundary=False, num_players=4):
    ess = tuple([N//num_players]*num_players)
    #m=[[0,1,1],[1,0,1],[1,1,0]]
    m=[[0,1,1,1],[1,0,1,1],[1,1,0,1],[1,1,1,0]]
    #m = rock_scissors_paper(a=1, b=-2)
    fitness_landscape = linear_fitness_landscape(m)
    # Approximate calculation
    incentive = logit(fitness_landscape, beta=1., q=1.)
    #edges = multivariate_moran_transitions(N, incentive, mu=mu)
    edges = incentive_process.multivariate_transitions(N, incentive, mu=mu, no_boundary=no_boundary, num_types=num_players)
    #d3 = approximate_stationary_distribution(N, edges, iterations=iterations)

    t = dict()
    for s1, s2, v in edges:
        t[(s1, s2)] = v

    s1 = 0.
    s2 = 0
    
    for a1, a2, v in edges:
        if a1 == a2:
            continue
        if a1 == ess:
            s1 += v
            print a1, t[(a1, ess)], t[(ess, a1)] 
        if a2 == ess:
            s2 += v
    print s1, s2

if __name__ == '__main__':
    transition_test()
    exit()
