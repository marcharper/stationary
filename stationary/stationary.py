from collections import defaultdict

from graph import Graph
from math_helpers import kl_divergence, simplex_generator, factorial, inc_factorial, log_factorial, log_inc_factorial

from numpy import array, log, exp, zeros

from scipy.special import gammaln
from scipy.misc import logsumexp

# numpy.seterr(all="print")

# Reference: http://www.statslab.cam.ac.uk/~frank/BOOKS/book/ch1.pdf

## Helpers

def enumerate_states(edges, inverse=True):
    """
    Enumerates the states of a Markov process from the list of edges.

    Parameters
    ----------
    edges: list of tuples
        Transition probabilities of the form [(source, target, transition_probability
    inverse: bool True
        Include the inverse enumeration

    Returns
    -------
    all_states, set
        The set of all states of the process
    enum, dict
        A dictionary mapping states to integers
    inv_enum, dict
        A dictionary mapping integers to states
    """

    # Collect the states
    all_states = set()
    for (source, target, weight) in edges:
        all_states.add(source)
        all_states.add(target)

    if not inverse:
        enum = dict(zip(all_states, range(len(all_states))))
        return (all_states, enum)

    # Enumerate the states and compute the inverse enumeration
    enum = dict()
    inv_enum = []
    for index, state in enumerate(all_states):
        enum[state] = index
        inv_enum.append(state)
    return (all_states, enum, inv_enum)

def edges_to_matrix(edges):
    """
    Converts a list of edges to a transition matrix by enumerating the states.

    Parameters
    ----------
    edges: list of tuples
        Transition probabilities is the form [(source, target, transition_probability

    Returns
    -------
    mat, numpy.array
        The transition matrix
    all_states, list of nodes
        The collection of states
    enumeration, dictionary
        maps states to integers
    """

    # Enumerate states so we can put them in a matrix.
    all_states, enumeration = enumerate_states(edges, inverse=False)

    # Build a matrix for the transitions
    mat = numpy.zeros((len(all_states), len(all_states)))
    for (a, b, v) in edges:
        mat[enumeration[a]][enumeration[b]] = v
    return mat, all_states, enumeration

def edges_to_edge_dict(edges):
    """
    Converts a list of edges to a transition dictionary taking (source, target)
    to the transition between the states.

    Parameters
    ----------
    edges: list of tuples
        Transition probabilities of the form [(source, target, transition_probability

    Returns
    -------
    edges, maps 2-tuples of nodes to floats
    """

    edge_dict = dict()
    for e1, e2, v in edges:
        edge_dict[(e1, e2)] = v
    return edge_dict

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


def stationary_distribution_generator(cache, initial_state=None):
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

    Yields
    ------
    a list of floats
    """

    N = len(cache.inv_enum)
    if not initial_state:
        ranks = [1/float(N)]*N
    else:
        ranks = initial_state
    # This is essentially iterated sparse matrix multiplication.
    yield ranks
    while True:
        new_ranks = []
        for node in range(N):
            new_rank = 0.
            for i, v in cache.in_neighbors[node]:
                new_rank += v * ranks[i]
            new_ranks.append(new_rank)
        yield new_ranks
        ranks = new_ranks

def log_stationary_distribution_generator(cache, initial_state=None):
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

    Yields
    ------
    a list of floats
    """

    N = len(cache.inv_enum)
    if not initial_state:
        ranks = [-log(float(N))]*N
    else:
        ranks = initial_state
    while True:
        new_ranks = []
        for node in range(N):
            l = []
            for i, v in cache.in_neighbors[node]:
                l.append(log(v) + ranks[i])
            new_ranks.append(logsumexp(l))
        yield exp(new_ranks)
        ranks = new_ranks

def output_enumerated_edges(N, n, edges, filename="enumerated_edges.csv"):
    """
    Writes the graph underlying to the Markov process to disk. This is used to
    export the computation to a C++ implementation if the number of nodes is
    very large.
    """

    # Collect all the states from the list of edges
    all_states, enum, inv_enum = enumerate_states(edges, inverse=True)

    # Output enumerated_edges
    with open(filename, 'w') as outfile:
        outfile.write(str(num_states(N,n)) + "\n")
        outfile.write(str(n) + "\n")
        for (source, target, weight) in edges:
            row = [str(enum[source]), str(enum[target]), str.format('%.50f' % weight)]
            outfile.write(",".join(row) + "\n")
    return inv_enum

### Exact computations for reversible processes. Use at your own risk! No check for reversibility is performed

def exact_stationary_distribution(edges, initial=None):
    """
    Computes the stationary distribution of a reversible process on the simplex exactly. No check for reversibility.

    Parameters
    ----------

    edges: list or dictionary
        The edges or edge_dict of the process
    initial: tuple, None
        The initial state. If not given a suitable state is created.

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
    if not initial:
        initial = [N//num_players]*(num_players)
        initial[-1] = N - (num_players-1) * (N//num_players)
    initial = tuple(initial)

    # Use the exact form of the stationary distribution.
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

def log_exact_stationary_distribution(edges, initial=None, no_boundary=False):
    """
    Same as the exact calculation in exact_stationary_distribution in
    log-space.

    Parameters
    ----------

    edges: list or dictionary
        The edges or edge_dict of the process
    initial: tuple, None
        The initial state. If not given a suitable state is created.

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

# Note: For the Wright-Fisher process use the direct implementation stationary
# function in wright_fisher.py

def approximate_stationary_distribution(edges, iterations=None, convergence_lim=1e-8):
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
    iterations: int, None
        Maximum number of iterations
    convergence_lim: float, 1e-13
        Approximate algorithm breaks when successive iterations have a
        KL-divergence less than convergence_lim
    """

    g = Graph()
    g.add_edges(edges)
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

    # Reverse the enumeration
    d = dict()
    for m, r in enumerate(ranks):
        state = cache.inv_enum[m]
        d[(state)] = r
    return d

def log_approximate_stationary_distribution(edges, iterations=None, convergence_lim=1e-9):
    """
    Approximate stationary distributions computed by by sparse matrix
    multiplications. Produces correct results and uses little memory but is
    likely not the most CPU efficient implementation in general (e.g. an
    eigenvector calculator may be better). This is a log-space version that is
    more accurate in general.

    Essentially raises the transition probabilities matrix to a large power.

    Parameters
    -----------
    edges: list of tuples
        Transition probabilities of the form [(source, target, transition_probability
    iterations: int, None
        Maximum number of iterations
    convergence_lim: float, 1e-13
        Approximate algorithm breaks when successive iterations have a
        KL-divergence less than convergence_lim
    """

    g = Graph()
    g.add_edges(edges)
    cache = Cache(g)
    gen = log_stationary_distribution_generator(cache)
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
    # Reverse enumeration
    d = dict()
    for m, r in enumerate(ranks):
        state = cache.inv_enum[m]
        d[(state)] = r
    return d

def compute_stationary(edges, exact=False, convergence_lim=1e-13):
    """
    Convenience Function for computing stationary distribution.

    Parameters
    ----------
    edges: list of tuples
        Transition probabilities of the form [(source, target, transition_probability
    exact: Bool, False
        Use exact or approximate algorithm to compute stationary distribution.
        Approximate works for any process, exact only for reversible processes
        on the simplex.
    convergence_lim: float, 1e-13
        Approximate algorithm breaks when successive iterations have a
        KL-divergence less than convergence_lim

    Returns
    -------
    dictionary, stationary distribution of the process
    """

    if not exact:
        # Approximate Calculation
        s = approximate_stationary_distribution(edges, convergence_lim=convergence_lim)
    else:
        # Exact Calculuation
        edge_dict = edges_to_edge_dict(edges)
        s = exact_stationary_distribution(edge_dict)
    return s

## Neutral landscape / Dirichlet

def neutral_stationary(N, alpha, n=3):
    """
    Computes the stationary distribution of the neutral landscape. This process
    is always reversible and there is an explicit formula.

    Parameters
    ----------
    N: int
        Population size / simplex divisor
    alpha:
        Parameter defining the stationary distribution in terms of n and mu
    n: int, 3
        Simplex dimension - 1, Number of types in population

    Returns
    -------
    dictionary, stationary distribution of the process
    """

    # Large N is better handled by the log version to avoid underflows
    if N > 100:
        return log_neutral_stationary(N, alpha, n=n)

    # Just compute the distribution directly for each state
    d2 = dict()
    for state in simplex_generator(N, n-1):
        t = 1.
        for i in state:
            t *= inc_factorial(alpha, i) / factorial(i)
        t *= factorial(N) / inc_factorial(n * alpha, N)        
        d2[state] = t
    return d2

def log_neutral_stationary(N, alpha, n=3):
    """
    Computes the stationary distribution of the neutral landscape. This process
    is always reversible and there is an explicit formula. This function is the 
    same as neutral_stationary in log-space.

    Parameters
    ----------
    N: int
        Population size / simplex divisor
    alpha:
        Parameter defining the stationary distribution in terms of n and mu
    n: int, 3
        Simplex dimension - 1, Number of types in population

    Returns
    -------
    dictionary, stationary distribution of the process
    """

    d2 = dict()
    for state in simplex_generator(N, n-1):
        t = 0.
        for i in state:
            t += log_inc_factorial(alpha, i) - log_factorial(i)
        t += log_factorial(N) - log_inc_factorial(n * alpha, N)
        d2[state] = exp(t)
    return d2

## Entropy Rate

def entropy_rate(edges, stationary):
    """
    Computes the entropy rate given the edges of the process and the stationary distribution.

    Parameters
    ----------
    edges: list of tuples
        Transition probabilities of the form [(source, target, transition_probability
    stationary: dictionary
        Precomputed stationary distribution

    Returns
    -------
    float, entropy rate of the process
    """

    e = defaultdict(float)
    for a,b,v in edges:
        e[a] -= stationary[a] * v * log(v)
    return sum(e.values())

def entropy_rate_func(N, edge_func, stationary):
    """
    Computes entropy rate for a process with a large transition matrix, defined
    by a transition function (edge_func) rather than a list of weighted edges.

    Use when the number of states or the transition matrix is prohibitively
    large, e.g. for the Wright-Fisher process.

    Parameters
    ----------
    N: int
        Population size / simplex divisor
    edge_func, function
        Yields the transition probabilities between two states, edge_func(a,b)
    stationary: dictionary
        Precomputed stationary distribution
    """

    e = defaultdict(float)
    for a in simplex_generator(N):
        for b in simplex_generator(N):
            v = edge_func(a,b)
            e[a] -= stationary[a] * v * log(v)
    return sum(e.values())
