from collections import defaultdict
from itertools import product
from operator import itemgetter
import sys

import numpy

from stationary.utils.graph import Graph
from stationary import stationary_distribution
from stationary.processes.graph_process import multivariate_graph_transitions
from stationary.processes.incentives import replicator, linear_fitness_landscape

def cycle(length, directed=False):
    """
    Produces a cycle of length `length`.

    Parameters
    ----------
    length: int
        Number of vertices in the cycle
    directed: bool, False
        Is the cycle directed?

    Returns
    -------
    a Graph object
    """

    graph = Graph()
    edges = []
    for i in range(length - 1):
        edges.append((i, i+1))
        if not directed:
            edges.append((i+1, i))
    edges.append((length - 1, 0))
    if not directed:
        edges.append((0, length - 1))
    graph.add_edges(edges)
    return graph

def cycle_configurations_consolidation(N):
    """
    Consolidates cycle configurations based on rotational symmetry.
    """
    config_id = 0
    config_mapper = dict()
    inverse_mapper = dict()

    for c in product([0, 1], repeat=N):
        if c not in config_mapper:
            # cycle through the list and add to the mapper
            for i in range(len(c)):
                b = numpy.roll(c,i)
                config_mapper[tuple(b)] = config_id
            # Record a representative
            inverse_mapper[config_id] = c
            config_id += 1
    return (config_mapper, inverse_mapper)

def consolidate_stationary(s, N):
    """
    Consolidates stationary distribution over cycle rotations.
    """
    config_mapper, inverse_mapper = cycle_configurations_consolidation(N)
    new_s = defaultdict(float)
    for k,v in s.items():
        new_s[config_mapper[k]] += v
    return new_s, inverse_mapper

def cycle_stationary_example(N, m, mu, incentive_func=replicator):
    graph = cycle(N)
    fitness_landscape = linear_fitness_landscape(m)
    incentive = incentive_func(fitness_landscape)

    edge_dict = multivariate_graph_transitions(N, graph, incentive, num_types=2, mu=mu)
    print "There are %s configurations and %s transitions" % (len(set([x[0] for x in edge_dict.keys()])), len(edge_dict))

    # Compute stationary distribution
    edges = [(v1, v2, t) for ((v1,v2),t) in edge_dict.items()]
    s = stationary_distribution(edges, lim=1e-8)
    return s

if __name__ == '__main__':
    try:
        N = int(sys.argv[1])
    except IndexError:
        N = 10
    try:
        mu = sys.argv[2]
    except IndexError:
        mu = 1./N
    m = [[1,2],[2,1]]
    s = cycle_stationary_example(N, m, mu, incentive_func=replicator)
    # Consolidate states
    s, inverse_mapper = consolidate_stationary(s, N)
    # Print stationary distribution top 20
    print "Stationary"
    for k,v in sorted(s.items(), key=itemgetter(1), reverse=True)[:20]:
        rep = inverse_mapper[k]
        print rep, sum(rep), v
