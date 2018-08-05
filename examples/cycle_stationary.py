from __future__ import print_function
from collections import defaultdict
from itertools import product
from operator import itemgetter
import sys

import numpy

from stationary.utils.graph import Graph
from stationary import stationary_distribution
from stationary.processes.graph_process import multivariate_graph_transitions
from stationary.processes.incentives import replicator, linear_fitness_landscape

import faulthandler
faulthandler.enable()


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
    return config_mapper, inverse_mapper


def consolidate_stationary(s, N):
    """
    Consolidates stationary distribution over cycle rotations.
    """
    config_mapper, inverse_mapper = cycle_configurations_consolidation(N)
    new_s = defaultdict(float)
    for k, v in s.items():
        new_s[config_mapper[k]] += v
    return new_s, inverse_mapper


def find_extrema_stationary(s, g, extrema="max"):
    extreme_states = []
    for state in g.vertices():
        is_extrema = True
        v = s[state]
        for neighbor in g.out_vertices(state):
            if state == neighbor:
                continue
            if extrema == "max" and s[neighbor] > v:
                is_extrema = False
                break
            elif extrema == "min" and s[neighbor] < v:
                is_extrema = False
                break
        if is_extrema:
            extreme_states.append(state)
    return extreme_states


def find_extrema_yen(graph, extrema="max"):
    extreme_states = []
    for state in g.vertices():
        is_extrema = True
        for neighbor in g.out_vertices(state):
            if state == neighbor:
                continue
            tout = g[state][neighbor]
            tin = g[neighbor][state]
            if extrema == "max" and (tout < tin):
                is_extrema = False
                break
            elif extrema == "min" and (tout > tin):
                is_extrema = False
                break
        if is_extrema:
            extreme_states.append(state)
    return extreme_states


if __name__ == '__main__':
    try:
        N = int(sys.argv[1])
    except IndexError:
        N = 10
    try:
        mu = sys.argv[2]
    except IndexError:
        mu = 1./N

    #m = [[1,1], [1,1]]
    #m = [[1,2], [2,1]]
    #m = [[2,1],[1,2]]
    #m = [[2,2],[2,1]]
    m = [[2,2],[1,1]]
    print(N, m, mu)

    graph = cycle(N)
    fitness_landscape = linear_fitness_landscape(m)
    incentive = replicator(fitness_landscape)
    edge_dict = multivariate_graph_transitions(N, graph, incentive, num_types=2, mu=mu)
    edges = [(v1, v2, t) for ((v1, v2), t) in edge_dict.items()]
    g = Graph(edges)

    print("There are %s configurations and %s transitions" % (len(set([x[0] for x in edge_dict.keys()])), len(edge_dict)))

    print("Local Maxima:", len(find_extrema_yen(g, extrema="max")))
    print("Local Minima:", len(find_extrema_yen(g, extrema="min")))
    print("Total States:", 2**N)

    exit()
    print("Computing stationary")
    s = stationary_distribution(edges, lim=1e-8, iterations=1000)
    print("Local Maxima:", len(find_extrema_stationary(s, g, extrema="max")))
    print("Local Minima:", len(find_extrema_stationary(s, g, extrema="min")))

    # Print stationary distribution top 20
    print("Stationary")
    for k, v in sorted(s.items(), key=itemgetter(1), reverse=True)[:20]:
        print(k, v)

    print(len([v for v in s.values() if v > 0.001]), sum([v for v in s.values() if v > 0.001]))

    # Consolidate states
    s, inverse_mapper = consolidate_stationary(s, N)
    # Print stationary distribution top 20
    print("Consolidated Stationary")
    for k,v in sorted(s.items(), key=itemgetter(1), reverse=True)[:20]:
        rep = inverse_mapper[k]
        print(rep, sum(rep), v)

    print(len([v for v in s.values() if v > 0.001]),
          sum([v for v in s.values() if v > 0.001]))

