from collections import defaultdict
from itertools import product

import numpy

from ..utils.graph import Graph


## Don't put N > 20 into this unless you have a lot of RAM and time
def multivariate_graph_transitions(N, graph, incentive, mu=0.001):
    """
    Computes transition probabilities of the incentive process on a graph.
    Warning: this uses a LOT of RAM (exponential in N typically), keep N small.

    Parameters
    ----------
    N: int
        Population size / simplex divisor
    graph: Graph
        The graph that the population occuupies
    incentive: function
        An incentive function from incentives.py
    mu: float, 0.001
        The mutation rate of the process
    """

    def population_state(N, config):
        """Calculates the population state from a graph configuration, i.e.
        if the population were well-mixed on a complete graph."""
        s = sum(config)
        population_state_ = (N-s, s)
        return numpy.array(population_state_)

    edges = defaultdict(float)

    # Enumerate the graph vertices
    enum = dict(enumerate(graph.vertices()))
    inv_enum = dict([(y, x) for (x, y) in enumerate(graph.vertices())])

    # Generate all binary strings (configurations)
    for source_config in product([0, 1], repeat=N):
        edges[(source_config, source_config)] = 1.
        # For each position in the configuration, mutate it and replace one of
        # its neighbors, as dictated by the graph.
        s = sum(source_config)
        population_state = (N - s, s)
        inc = incentive(population_state)
        denom = float(sum(inc))
        for source_position, source_type in enumerate(source_config):
            # Probability that this one was picked to reproduce is
            r = 1. / population_state[source_type] * float(inc[source_type]) / denom
            # Replace out neighbors
            source_vertex = inv_enum[source_position]
            out_vertices = graph.out_vertices(source_vertex)
            total_out_vertices = float(len(out_vertices))
            for target_vertex in graph.out_vertices(source_vertex):
                target_position = enum[target_vertex]
                # Replace without mutation
                target_config = list(source_config)
                target_config[target_position] = source_type
                target_config = tuple(target_config)
                if source_config != target_config:
                    t = r * (1. - mu) / total_out_vertices
                    edges[(source_config, target_config)] += t
                    edges[(source_config, source_config)] -= t
                # Replace with mutation
                target_config = list(source_config)
                target_config[target_position] = 1 - source_type
                target_config = tuple(target_config)
                if source_config != target_config:
                    t = r * mu / total_out_vertices
                    edges[(source_config, target_config)] += t
                    edges[(source_config, source_config)] -= t
    return edges

