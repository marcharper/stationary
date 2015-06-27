from numpy import zeros
from numpy.linalg import matrix_power

from math_helpers import simplex_generator

def states_from_edges(edges):
    """
    Computes the underlying set of states from the list of edges.

    Parameters
    ----------
    edges: list of tuples
        Transition probabilities of the form [(source, target, transition_probability

    Returns
    -------
    states, set
        The set of all the vertices of the edge list
    """
    states = set()
    for (source, target, weight) in edges:
        states.add(source)
        states.add(target)
    return states

def enumerate_states(states, inverse=True):
    """
    Enumerates a list of states, and possibly with the inverse mapping.

    Parameters
    ----------
    states: List
        The list of hashable objects to enumerate
    inverse: bool True
        Include the inverse enumeration

    Returns
    -------
    enum, dict
        A dictionary mapping states to integers
    inv_enum, list
        A list mapping integers to states
    """

    if not inverse:
        enum = dict(zip(states, range(len(states))))
        return enum

    enum = dict()
    inv_enum = []
    for i, state in enumerate(states):
        enum[state] = i
        inv_enum.append(state)
    return (enum, inv_enum)

def enumerate_states_from_edges(edges, inverse=True):
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
    inv_enum, list
        A list mapping integers to states
    """

    # Collect the states
    all_states = states_from_edges(edges)

    if not inverse:
        enum = enumerate_states(all_states, inverse=False)
        return (all_states, enum)

    enum, inv_enum = enumerate_states(all_states, inverse=True)
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
    all_states, enumeration = enumerate_states_from_edges(edges, inverse=False)

    # Build a matrix for the transitions
    mat = zeros((len(all_states), len(all_states)))
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

def output_enumerated_edges(N, n, edges, filename="enumerated_edges.csv"):
    """
    Writes the graph underlying to the Markov process to disk. This is used to
    export the computation to a C++ implementation if the number of nodes is
    very large.
    """

    # Collect all the states from the list of edges
    all_states, enum, inv_enum = enumerate_states_from_edges(edges, inverse=True)

    # Output enumerated_edges
    with open(filename, 'w') as outfile:
        outfile.write(str(num_states(N,n)) + "\n")
        outfile.write(str(n) + "\n")
        for (source, target, weight) in edges:
            row = [str(enum[source]), str(enum[target]), str.format('%.50f' % weight)]
            outfile.write(",".join(row) + "\n")
    return inv_enum

def edge_func_to_edges(edge_func, states):
    """
    Convert an edge_func to a list of edges.
    """

    edges = []
    for s1 in states:
        for s2 in states:
            edges.append((s1, s2, edge_func(s1, s2)))
    return edges

def power_transitions(edges, k=20):
    """
    Raises a transition matrix (specified by edges) to the power `k`, returning
    an edge_func.
    """

    mat, all_states, enum = edges_to_matrix(edges)
    M_k = matrix_power(mat, k)
    def edge_func(current_state, next_state):
        return M_k[enum[current_state]][enum[next_state]]
    return edge_func
