"""
Labeled and weighted graph class for Markov simulations. Not a full-featured
class, rather an appropriate organizational data structure for handling various
Markov process calculations.
"""

from collections import defaultdict
import random


class Graph(object):
    """Directed graph object intended for the graph associated to a Markov process.
    Gives easy access to the neighbors of a particular state needed for various
    calculations.

    Vertices can be any hashable / immutable python object.
    """

    def __init__(self, edges=None):
        self.out_mapping = defaultdict(lambda: defaultdict(float))
        self.in_mapping = defaultdict(lambda: defaultdict(float))
        if edges:
            self.add_edges(edges)

    def add_vertex(self, label):
        self._vertices.add(label)

    def add_edge(self, source, target, weight=1.):
        self.out_mapping[source][target] = weight
        self.in_mapping[target][source] = weight

    def add_edges(self, edges):
        try:
            for source, target, weight in edges:
                self.add_edge(source, target, weight)
        except ValueError:
            for source, target in edges:
                self.add_edge(source, target, 1.0)

    def vertices(self):
        """Returns the set of vertices of the graph."""
        return self.out_mapping.keys()

    def out_dict(self, source):
        """Returns a dictionary of the outgoing edges of source with weights."""
        return self.out_mapping[source]

    def out_vertices(self, source):
        """Returns a list of the outgoing vertices."""
        return self.out_mapping[source].keys()

    def in_dict(self, target):
        """Returns a dictionary of the incoming edges of source with weights."""
        #return dict([(s, v) for (s, t, v) in self._edges if t == target])
        return self.in_mapping[target]

    def normalize_weights(self):
        """Normalizes the weights coming out of each vertex to be probability
        distributions."""
        new_edges = []
        for source in self.out_mapping.keys():
            total = float(sum(out_mapping[source].values()))
            #out_edges = [(s, t, v) for s, t, v in self._edges if s == source]
            #total = sum(v for (s, t, v) in out_edges)
            #for s, t, v in out_edges:
                #new_edges.append((s,t,v/total))
            for target, weight in self.out_mapping.items():
                self.out_mapping[target] = weight / total
        self._edges = new_edges

        #self.out_mapping = defaultdict(lambda: defaultdict(float))
        #self.in_mapping = defaultdict(lambda: defaultdict(float))



    #def normalize_weights(self):
        #"""Normalizes the weights coming out of each vertex to be probability
        #distributions."""
        #new_edges = []
        #for source in self.vertices():
            #out_edges = [(s, t, v) for s, t, v in self._edges if s == source]
            #total = sum(v for (s, t, v) in out_edges)
            #for s, t, v in out_edges:
                #new_edges.append((s,t,v/total))
        #self._edges = new_edges

    def right_multiply(self, d):
        """
        Multiply by a vector (specified as a dict on the vertices) by 
        viewing the graph as a sparse matrix, i.e.
        return G*d
        """
        s = defaultdict(float)
        for k in d.keys():
            for k2, v2 in self.out_dict(k).items():
                s[k] += d[k2] * v2
        return s

    def left_multiply(self, d):
        """
        Multiply by a vector (specified as a dict on the vertices) by 
        viewing the graph as a sparse matrix, i.e.
        return d*G
        """
        s = defaultdict(float)
        for k in d.keys():
            for k2, v2 in self.in_dict(k).items():
                s[k] += d[k2] * v2
        return s

    def __getitem__(self, k):
        """Returns the dictionary of outgoing edges. You can access the weight
        of an edge with g[source][target]."""
        return self.out_mapping[k]


class RandomGraph(object):
    """Random Graph class in which there is a probability p of an edge between any two vertices. Edge existence is drawn on each request (i.e. not determined once at initiation)."""
    def __init__(self, num_vertices, p):
        self._vertices = list(range(num_vertices))
        self.p = p

    def vertices(self):
        return self._vertices

    def out_vertices(self, source):
        outs = []
        for v in self._vertices:
            q = random.random()
            if q <= self.p:
                outs.append(v)
        return outs

    def in_vertices(self, source):
        ins = []
        for v in self._vertices:
            q = random.random()
            if q <= self.p:
                ins.append(v)
        return ins
