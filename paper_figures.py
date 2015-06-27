import math
import os
from matplotlib import pyplot

from stationary import stationary_distribution
from stationary.processes.incentives import *
from stationary.processes import incentive_process
from stationary.utils.math_helpers import simplex_generator, q_divergence

import ternary

#def bomze_figures(N=80, indices=[2,20,47], beta=1, process="incentive", directory=None):
    #if not directory:
        #directory = "bomze_paper_figures_%s" % process
        #if not os.path.isdir(directory):
            #os.mkdir(directory)
    #from three_dim import m_gen, bomze_plots
    #for i, m in enumerate(m_gen()):
        #if i not in indices:
            #continue
        #print i
        #mu = 3./2 * 1./N
        ##mu = (1./2)*1./N
        #bomze_plots(N=N, m=m, mu=mu, i=i, directory=directory, beta=beta,
                    #q_ds=(0., 0.5, 1.0), process=process)


def slice_dictionary(d, N, slice_index=0, slice_value=0):
    """
    Take a three dimensional slice from a four dimensional
    dictionary.
    """

    slice_dict = dict()
    for state in simplex_generator(N, 2):
        new_state = list(state)
        new_state.insert(slice_index, slice_value)
        slice_dict[state] = d[tuple(new_state)]
    return slice_dict

def four_dim_figures(N=30, beta=1., q=1.):
    """
    Four dimensional example. Three dimensional slices are plotted
    for illustation.
    """
    m = [[0, 1, 1, 1], [1, 0, 1, 1], [1, 1, 0, 1], [0,0,0,1]]
    num_types = len(m[0])
    fitness_landscape = linear_fitness_landscape(m)
    mu = 4. / 3 * 1. / N

    incentive = fermi(fitness_landscape, beta=beta, q=q)
    edges = incentive_process.multivariate_transitions(N, incentive, num_types=num_types, mu=mu)

    d1 = incentive_process.kl(edges, q_d=0, boundary=True)
    d2 = stationary_distribution(edges)

    # We need to slice the 4dim dictionary into three-dim slices for plotting.
    for slice_index in range(4):
        for d in [d1, d2]:
            slice_dict = slice_dictionary(d, N, slice_index=3)
            #temp = dict()
            #slice_val = 0
            #for state in simplex_generator(N - slice_val, 2):
                #a, b, c = state
                ##full_state = (a,b,c, slice_val)
                ##full_state = (0, a,b,c)
                #full_state = (a, b, 0, c)
                #temp[state] = d[full_state]
            figure, tax = ternary.figure(scale=N)
            tax.heatmap(slice_dict, style="d")

    pyplot.show()



## Graphical Abstract ## 

def graphical_abstract_figures(N=60, q=1, beta=0.1):
    """
    Three dimensional process examples.
    """

    a = 0
    b = 1
    m = [[a, b, b], [b, a, b], [b, b, a]]
    mu = (3. / 2 ) * 1. / N
    fitness_landscape = linear_fitness_landscape(m)
    incentive = fermi(fitness_landscape, beta=beta, q=q)
    edges = incentive_process.multivariate_transitions(N, incentive, num_types=3, mu=mu)
    d = stationary_distribution(edges, iterations=None)

    figure, tax = ternary.figure(scale=N)
    tax.heatmap(d, scale=N)
    tax.savefig(filename="ga_stationary.eps", dpi=600)

    d = incentive_process.kl(edges, q_d=0)
    figure, tax = ternary.figure(scale=N)
    tax.heatmap(d, scale=N)
    tax.savefig(filename="ga_d_0.eps", dpi=600)

    d = incentive_process.kl(edges, q_d=1)
    figure, tax = ternary.figure(scale=N)
    tax.heatmap(d, scale=N)
    tax.savefig(filename="ga_d_1.eps", dpi=600)

def rps_figures(N=60, q=1, beta=1.):
    """
    Three rock-paper-scissors examples.
    """

    m = [[0, -1, 1], [1, 0, -1], [-1, 1, 0]]
    num_types = len(m[0])
    fitness_landscape = linear_fitness_landscape(m)
    for i, mu in enumerate([1./math.sqrt(N), 1./N, 1./N**(3./2)]):
        # Approximate calculation
        mu = 3/2. * mu
        incentive = fermi(fitness_landscape, beta=beta, q=q)
        edges = incentive_process.multivariate_transitions(N, incentive, num_types=num_types, mu=mu)
        d = stationary_distribution(edges, lim=1e-10)

        figure, tax = ternary.figure()
        tax.heatmap(d, scale=N)
        tax.savefig(filename="rsp_mu_" + str(i) + ".eps", dpi=600)

def tournament_stationary(N, mu=None):
    """
    Example for a tournament selection matrix.
    """

    if not mu:
        mu = 3./2 * 1./N
    m = [[1,1,1], [0,1,1], [0,0,1]]
    num_types = len(m[0])
    fitness_landscape = linear_fitness_landscape(m)
    incentive = replicator(fitness_landscape)
    edges = incentive_process.multivariate_transitions(N, incentive, num_types=num_types, mu=mu)
    s = stationary_distribution(edges)
    ternary.heatmap(s, scale=N, scientific=True)
    d = incentive_process.kl(edges, q_d=0)
    ternary.heatmap(d, scale=N, scientific=True)
    pyplot.show()

if __name__ == '__main__':

    ## Four dimensional examples
    four_dim_figures()
    exit()

    ## Three dimensional examples
    graphical_abstract_figures()
    rps_figures()

    ## Tournament Selection 
    N = 60
    m = 1. / 3
    tournament_stationary(N, mu=mu)



