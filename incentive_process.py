from scipy.misc import logsumexp
from math_helpers import kl_divergence, simplex_generator, one_step_indicies_generator, dot_product, normalize, q_divergence

import numpy
from numpy import array, log, exp

from incentives import *

########################
### 3d Moran process ###
########################

def is_valid_state(N, state, lower, upper):
    for i in state:
        if i < lower or i > upper:
            return False
    return True

def kl(N, edges, q_d=1, boundary=False):
    """Computes the KL-div of the expected state with the state, for all states."""
    dist = q_divergence(q_d)
    e = dict()
    for x, y, w in edges:
        try:
            e[x] += numpy.array(y) * w
        except KeyError:
            e[x]  = numpy.array(y) * w
    d = dict()
    for state, v in e.items():
        # KL doesn't play well on the boundary.
        if not boundary:
            p = 1.
            for s in state:
                p *= s
            if p == 0:
                continue
        d[state] = dist(normalize(v), normalize(list(state)))
    return d

def multivariate_transitions(N, incentive, num_types=3, mu=0.001, no_boundary=False):
    """Computes transitions for dimension n=3 moran process given a game matrix."""
    d = num_types - 1
    edges = []
    one_step_indicies = list(one_step_indicies_generator(d))
    if no_boundary:
        lower, upper = 1, N-1
    else:
        lower, upper = 0, N
    for state in simplex_generator(N, d):
        if no_boundary:
            is_boundary = False
            for i in state:
                if i == 0:
                    is_boundary = True
                    break
            if is_boundary:
                continue
        s = 0.
        inc = incentive(state)
        denom = float(sum(inc))
        # Transition probabilities for each adjacent state.
        for plus_index, minus_index in one_step_indicies:
            target_state = list(state)
            target_state[plus_index] += 1
            target_state[minus_index] -= 1
            target_state = tuple(target_state)
            # Is this a valid state? I.E. Are we on or near the boundary?
            if not is_valid_state(N, target_state, lower, upper):
                continue
            #mutations = [mu] * num_types
            #mutations[plus_index] = 1. - d*mu
            mutations = [mu / d] * num_types
            mutations[plus_index] = 1. - mu
            r = dot_product(inc, mutations) / denom
            transition = r * state[minus_index] / float(N)
            edges.append((state, target_state, transition))
            s += transition
        # Add in the transition probability for staying put.
        edges.append((state, state, 1. - s))
    return edges

def log_multivariate_transitions(N, logincentive, num_types=3, mu=0.001, no_boundary=False):
    """Computes the log of the transitions for dimension n=3 moran process given a game matrix."""
    d = num_types - 1
    edges = []
    one_step_indicies = list(one_step_indicies_generator(d))
    if no_boundary:
        lower, upper = 1, N-1
    else:
        lower, upper = 0, N
    for state in simplex_generator(N, d):
        if no_boundary:
            is_boundary = False
            for i in state:
                if i == 0:
                    is_boundary = True
                    break
            if is_boundary:
                continue
        inc = logincentive(state)
        denom = logsumexp(inc)
        # Transition probabilities for each adjacent state.
        logtransitions = []
        for plus_index, minus_index in one_step_indicies:
            target_state = list(state)
            target_state[plus_index] += 1
            target_state[minus_index] -= 1
            target_state = tuple(target_state)
            # Is this a valid state? I.E. Are we on or near the boundary?
            if not is_valid_state(N, target_state, lower, upper):
                continue
            mutations = [mu] * num_types
            mutations[plus_index] = 1. - d*mu
            r = logsumexp(inc, b=mutations) - denom
            logtransition = r + log(state[minus_index]) - log(N)
            edges.append((state, target_state, logtransition))
            logtransitions.append(logtransition)
        edges.append((state, state, log(1.-exp(logsumexp(logtransitions)))))
    return edges

######################
# Stationary Helpers #
######################

def compute_edges(N=30, num_types=2, m=None, incentive_func=logit, beta=1., q=1., mu=None):
    """Calculates the weighted edges of the incentive process."""
    if not m:
        m = numpy.ones((n,n))
    if not num_types:
        num_types = len(m[0])
    fitness_landscape = linear_fitness_landscape(m)
    incentive = incentive_func(fitness_landscape, beta=beta, q=q)
    if not mu:
        #mu = (n-1.)/n * 1./(N+1)
        mu = 1./(N)
    edges = multivariate_transitions(N, incentive, num_types=num_types, mu=mu)
    return edges
