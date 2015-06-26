"""
Stationary distributions and Entropy rates.
"""

from __future__ import absolute_import

from collections import defaultdict, Callable
import itertools

from numpy import log, exp, zeros

from stationary.utils.math_helpers import kl_divergence, simplex_generator, factorial, inc_factorial, log_factorial, log_inc_factorial, logsumexp, squared_error, squared_error_dict

from stationary.utils.graph import Graph

from stationary.utils.edges import *

def stationary_distribution(edges=None, exact=False, logspace=False, initial_state=None, iterations=None, lim=1e-12, states=None):
    """
    Convenience function to route to different stationary distribution
    computations.

    Parameters
    ----------
    logspace: bool False
        Carry out the calculation in logspace

    """
    if isinstance(edges, list):
        if not exact:
            return approx_stationary(edges, logspace=logspace,
                                        iterations=iterations, lim=lim,
                                        initial_state=initial_state)
        else:
            return exact_stationary(edges, initial_state=initial_state,
                                        logspace=logspace)
    elif isinstance(edges, Callable):
        return approx_stationary_func(edges, states, iterations=iterations, lim=lim)
    else:
        raise Exception, "Parameter combination not implemented"


## Stationary Distributions

class Cache(object):
    """
    Caches common calculations for a given graph associated to a Markov process
    for efficiency when computing the stationary distribution with the
    approximate algorithm.

    Parameters
    ----------
    graph: a Graph object
        The graph underlying the Markov process.
    """

    def __init__(self, graph):
        # Caches vertex enumeration, cumulative sums, absorbing state tests,
        # and transition targets.
        self.enum = dict()
        self.inv_enum = []
        self.in_neighbors = []
        self.terminals = []
        vertices = graph.vertices()
        # Enumerate vertices
        for (index, vertex) in enumerate(vertices):
            self.enum[vertex] = index
            self.inv_enum.append(vertex)
        # Cache in_neighbors
        for vertex in vertices:
            in_dict = graph.in_dict(vertex)
            self.in_neighbors.append([(self.enum[k], v) for k,v in 
                                      in_dict.items()])


def stationary_generator(cache, logspace=False, initial_state=None):
    """
    Generator for the stationary distribution of a Markov chain, produced by
    iteration of the transition matrix. The iterator yields successive
    approximations of the stationary distribution.

    Parameters
    ----------
    cache, a Cache object
    initial_state, None
        A distribution over the states of the process. If None, the uniform
        distiribution is used.
    logspace: bool False
        Carry out the calculation in logspace

    Yields
    ------
    a list of floats
    """

    N = len(cache.inv_enum)

    if logspace:
        sum_func = logsumexp
        exp_func = exp
        if not initial_state:
            initial_state = [-log(N)] * N
    else:
        sum_func = sum
        exp_func = lambda x: x
        if not initial_state:
            initial_state = [1. / N] * N

    ranks = initial_state

    # This is essentially iterated sparse matrix multiplication.
    yield ranks
    while True:
        new_ranks = []
        for node in range(N):
            l = []
            for i, v in cache.in_neighbors[node]:
                if logspace:
                    l.append(log(v) + ranks[i])
                else:
                    l.append(v * ranks[i])
            new_rank = sum_func(l)
            new_ranks.append(new_rank)
        ranks = new_ranks
        yield exp_func(ranks)

## Approximate stationary distributions computed by by sparse matrix multiplications.

def approx_stationary(edges, logspace=False, iterations=None, lim=1e-8,
                      initial_state=None):
    """
    Approximate stationary distributions computed by by sparse matrix
    multiplications. Produces correct results and uses little memory but is
    likely not the most CPU efficient implementation in general (e.g. an
    eigenvector calculator may be better).

    Essentially raises the transition probabilities matrix to a large power.

    Parameters
    -----------
    edges: list of tuples
        Transition probabilities of the form [(source, target, transition_probability
    logspace: bool False
        Carry out the calculation in logspace
    iterations: int, None
        Maximum number of iterations
    lim: float, 1e-13
        Approximate algorithm breaks when successive iterations have a
        KL-divergence less than lim
    """

    g = Graph()
    g.add_edges(edges)
    cache = Cache(g)
    gen = stationary_generator(cache, logspace=logspace, 
                               initial_state=initial_state)

    previous_ranks = None
    for i, ranks in enumerate(gen):
        if i > 200:
            if i % 10:
                s = squared_error(ranks, previous_ranks)
                if s < lim:
                    break
        if iterations:
            if i == iterations:
                break
        previous_ranks = ranks

    # Reverse the enumeration
    d = dict()
    for m, r in enumerate(ranks):
        state = cache.inv_enum[m]
        d[(state)] = r
    return d

def approx_stationary_func(edge_func, states, iterations=100, lim=1e-8):
    """
    Approximate stationary distributions computed by by sparse matrix
    multiplications. Produces correct results and uses little memory but is
    likely not the most CPU efficient implementation in general (e.g. an
    eigenvector calculator may be better).

    Essentially raises the transition probabilities matrix to a large power.

    This function takes a function that computes transitions rather than a list
    of edges, to lower the memory footprint (at the cost of efficiency). Needed
    for Wright-Fisher.

    Parameters
    -----------
    edge_func, function
        Yields the transition probabilities between two states, edge_func(a,b)
    iterations: int, None
        Maximum number of iterations
    lim: float, 1e-13
        Approximate algorithm breaks when successive iterations have a
        KL-divergence less than lim
    """

    ranks = dict(zip(states, [1./float(len(states))]*(len(states))))
    for iteration in itertools.count(1):
        if iterations:
            if iteration > iterations:
                break
        if iteration > 100:
            if iteration % 50:
                s = squared_error_dict(ranks, previous_ranks)
                if s < lim:
                    break
        new_ranks = dict()
        for x in states:
            new_rank = 0
            for y in states:
                w = edge_func(y,x)
                new_rank += w * ranks[y]
            new_ranks[x] = new_rank
        previous_ranks = ranks
        ranks = new_ranks
    d = dict()
    for m, r in ranks.items():
        d[m] = r
    return d


### Exact computations for reversible processes. Use at your own risk! No check for reversibility is performed

def exact_stationary(edges, initial_state=None, logspace=False):
    """
    Computes the stationary distribution of a reversible process on the simplex exactly. No check for reversibility.

    Parameters
    ----------

    edges: list or dictionary
        The edges or edge_dict of the process
    initial: tuple, None
        The initial state. If not given a suitable state is created.
    logspace: bool False
        Carry out the calculation in logspace

    returns
    -------
    dictionary, the stationary distribution
    """

    # Convert edges to edge_dict if necessary
    if isinstance(edges, list):
        edges = edges_to_edge_dict(edges)
    # Compute population parameters from the edge_dict
    state = edges.keys()[0][0]
    N = sum(state)
    num_players = len(state)
    # Get an initial state
    if not initial_state:
        initial_state = [N//num_players]*(num_players)
        initial_state[-1] = N - (num_players-1) * (N//num_players)
    initial_state = tuple(initial_state)

    # Use the exact form of the stationary distribution.
    d = dict()
    for state in simplex_generator(N, num_players-1):
        # Take a path from initial to state.
        seq = [initial_state]
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
        if logspace:
            s = 0.
        else:
            s = 1.
        for index in range(len(seq)-1):
            e, f = seq[index], seq[index+1]
            if logspace:
                s += log(edges[(e,f)]) - log(edges[(f, e)])
            else:
                s *= edges[(e,f)] / edges[(f, e)]
        d[state] = s
    if logspace:
        s0 = logsumexp([v for v in d.values()])
        for key, v in d.items():
            d[key] = exp(v-s0)
    else:
        s0 = 1./(sum([v for v in d.values()]))
        for key, v in d.items():
            d[key] = s0 * v
    return d
