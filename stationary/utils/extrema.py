import math
import numpy

from math_helpers import one_step_generator

def find_local_minima(d, comp_func=None):
    """
    Finds local minima of distributions on the simplex.

    Parameters
    ----------
    d: dict
        The dictionary on a simplex discretization to find the extrema of
    dim: int, 2
        The dimension of the simplex
    extremum: string, "min"
        Look for 'min' or 'max'

    Returns
    -------
    set of minimal states.
    """

    if not comp_func:
        comp_func = lambda x,y: (x - y >= 0)

    dim = len(d.keys()[0]) - 1
    states = []
    for state, value in d.items():
        if value is None:
            continue
        if math.isnan(value):
            continue
        is_extremum = True
        for one_step in one_step_generator(dim):
            adj = tuple(numpy.array(state) + numpy.array(one_step))
            try:
                v2 = d[adj]
            except KeyError:
                #print "KeyError", adj
                continue
            if comp_func(value, v2):
            #if value > v2:
                is_extremum = False
                break

        if is_extremum:
            states.append(state)
    return set(states)

def find_local_maxima(d):
    """
    Finds local maxima of distributions on the simplex.

    Parameters
    ----------
    d: dict
        The dictionary on a simplex discretization to find the extrema of
    dim: int, 2
        The dimension of the simplex
    extremum: string, "min"
        Look for 'min' or 'max'

    Returns
    -------
    set of maximal states.
    """

    comp_func = lambda x,y: (y - x >= 0)
    return find_local_minima(d, comp_func=comp_func)


def local_min_check(N=30, m=[[0,1,1],[1,0,1],[1,1,0]], beta=1., q=1., mu=0.001, iterations=None, process="incentive", plot_boundary=True, no_boundary=False):
    """This is a test function for verifying the main theorem."""
    num_types = len(m[0])
    fitness_landscape = linear_fitness_landscape(m)

    # Approximate calculation
    incentive = fermi(fitness_landscape, beta=beta, q=q)
    edges = incentive_process.multivariate_transitions(N, incentive, num_types=num_types, mu=mu)
    
    d = incentive_process.kl(N, edges, q=0, boundary=True)
    #d = incentive_process.kl(N, edges, q=1, boundary=True)
    d = incentive_process.kl(N, edges, q=1, boundary=False)
    local_mins = find_local_extrema(d, dim=num_types-1, extremum="min")
    
    d = approximate_stationary_distribution(N, edges, iterations=iterations)
    local_maxes = find_local_extrema(d, dim=num_types-1, extremum="max")

    print list(sorted(local_maxes))
    print list(sorted(local_mins))

def probability_neutral_check(N=30, m=[[0,1,1],[1,0,1],[1,1,0]], beta=1., q=1., mu=0.001, iterations=None, process="incentive", plot_boundary=True, no_boundary=False):
    """This is a test function for verifying the main theorem."""
    num_types = len(m[0])
    fitness_landscape = linear_fitness_landscape(m)

    # Approximate calculation
    incentive = fermi(fitness_landscape, beta=beta, q=q)
    edges = incentive_process.multivariate_transitions(N, incentive, num_types=num_types, mu=mu)

    e = probability_difference(N, edges)
    #heatmap(e)
    #pyplot.show()
    d = incentive_process.kl(N, edges, q_d=0, boundary=True)
    #d = incentive_process.kl(N, edges, q_d=1, boundary=False)
    local_mins = find_local_extrema(d, dim=num_types-1, extremum="min")
    pyplot.figure()
    heatmap(d)
    
    d = approximate_stationary_distribution(N, edges, iterations=iterations)
    local_maxes = find_local_extrema(d, dim=num_types-1, extremum="max")
    pyplot.figure()
    heatmap(d)

    pyplot.figure()
    heatmap(e)

    local_mins_2 = find_local_extrema(e, dim=num_types-1, extremum="min")
    print "stationary max", list(sorted(local_maxes))
    print "dist min", list(sorted(local_mins))
    print "prob flow", list(sorted(local_mins_2))