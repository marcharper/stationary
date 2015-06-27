from collections import defaultdict
import math
import os
import matplotlib
from matplotlib import pyplot
import matplotlib.gridspec as gridspec

from stationary import stationary_distribution
from stationary.processes.incentives import *
from stationary.processes import incentive_process, wright_fisher
from stationary.utils.math_helpers import simplex_generator, q_divergence, slice_dictionary
from stationary.utils.edges import edges_to_edge_dict
from stationary.utils.plotting import plot_stationary
from stationary.utils import expected_divergence

import ternary

# Font config for plots
font = {'size': 22}
matplotlib.rc('font', **font)

#def bomze_plots(N=40, m=None, i=0, directory="plots", beta=1., q=1., q_ds=None, mu=0.001, iterations=100, dpi=200, process="incentive", boundary=False):
    ##if not os.path.isdir(directory):
        ##os.mkdir(directory)
    #print process, i
    #fitness_landscape = linear_fitness_landscape(m)
    #if process == "incentive":
        #incentive = fermi(fitness_landscape, beta=beta, q=q)
        #edges = incentive_process.multivariate_transitions(N, incentive, num_types=3, mu=mu)
        #d = approximate_stationary_distribution(N, edges, iterations=iterations)
    #elif process == "wright_fisher":
        #incentive = fermi(fitness_landscape, beta=beta, q=q)
        #edge_func = wright_fisher.multivariate_transitions(N, incentive, mu=mu)
        #d = wright_fisher.stationary_distribution(N, edge_func, iterations=iterations)
    #print process, i, "stationary heatmap"
    #filename = os.path.join(directory, "%s_%s_stationary.eps" % (i, N))
    #heatmap(d, filename=filename, boundary=boundary)    
    #if not q_ds:
        #q_ds = [1.]
    #for q_d in q_ds:
        #if process == "incentive":
            #d = incentive_process.kl(N, edges, q_d=q_d)
        #elif process == "wright_fisher":
            #d = wright_fisher.kl(N, edge_func, q_d=q_d)
        #print process, i, "heatmap", q_d
        #filename = os.path.join(directory, "%s_%s_%s.eps"  % (i, N, q_d))
        #heatmap(d, filename=filename, boundary=boundary)   

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

    d1 = expected_divergence(edges, q_d=0, boundary=True)
    d2 = stationary_distribution(edges)

    # We need to slice the 4dim dictionary into three-dim slices for plotting.
    for slice_index in range(4):
        for d in [d1, d2]:
            slice_dict = slice_dictionary(d, N, slice_index=3)
            figure, tax = ternary.figure(scale=N)
            tax.heatmap(slice_dict, style="d")
    pyplot.show()

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

    d = expected_divergence(edges, q_d=0)
    figure, tax = ternary.figure(scale=N)
    tax.heatmap(d, scale=N)
    tax.savefig(filename="ga_d_0.eps", dpi=600)

    d = expected_divergence(edges, q_d=1)
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

def tournament_stationary_3(N, mu=None):
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
    d = expected_divergence(edges, q_d=0)
    ternary.heatmap(d, scale=N, scientific=True)
    pyplot.show()

def two_dim_transitions(edges):
    # Grab the transitions for plotting

    d = edges_to_edge_dict(edges)
    N = sum(edges[0][0])
    ups = []
    downs = []
    stays = []
    for i in range(0, N+1):
        try:
            up = d[((i,N-i), (i+1, N-i-1))]
        except KeyError:
            up = 0
        try:
            down = d[((i,N-i), (i+1, N-i-1))]
        except KeyError:
            down = 0
        ups.append(up)
        downs.append(down)
        stays.append(1 - up - down)
    return ups, downs, stays

def two_dim_transitions_figure(N, m, mu=0.01, incentive_func=replicator):
    """Plot transition entropies and stationary distributions."""
    n = len(m[0])
    fitness_landscape = linear_fitness_landscape(m)
    incentive = incentive_func(fitness_landscape)
    if not mu:
        mu = 1./ N
    edges = incentive_process.multivariate_transitions(N, incentive, num_types=n, mu=mu)

    s = stationary_distribution(edges, exact=True)
    d = edges_to_edge_dict(edges)

    # Set up plots
    gs = gridspec.GridSpec(3, 1)
    ax1 = pyplot.subplot(gs[0, 0])
    ax1.set_title("Transition Probabilities")
    ups, downs, _ = two_dim_transitions(edges)
    xs = range(0, N+1)
    ax1.plot(xs, ups)
    ax1.plot(xs, downs)

    ax2 = pyplot.subplot(gs[1, 0])
    ax2.set_title("Relative Entropy")
    divs1 = expected_divergence(edges)
    divs2 = expected_divergence(edges, q_d=0)
    plot_stationary(divs1, ax=ax2)
    plot_stationary(divs2, ax=ax2)

    ax3 = pyplot.subplot(gs[2, 0])
    ax3.set_title("Stationary Distribution")
    plot_stationary(s, ax=ax3)
    ax3.set_xlabel("Number of A individuals (i)")

#def two_dim_wright_fisher_figure(N, m, mu=0.01, incentive_func=replicator):
    #"""Plot transition entropies and stationary distributions."""
    #n = len(m[0])
    #fitness_landscape = linear_fitness_landscape(m)
    #incentive = incentive_func(fitness_landscape)
    #if not mu:
        #mu = 1./ N

    #wf_edges = wright_fisher.multivariate_transitions(N, incentive, mu=mu, num_types=2)
    #s = stationary_distribution(wf_edges, lim=1e-9)

    #d = edges_to_edge_dict(edges)

    #xs = range(0, N+1)
    #expected_states = []
    #for i in xs:
        #x_a = 0.
        #x_b = 0
        #for j in range(0, N+1):
            #x_a += j * d[(i, N-i)][(j, N-j)]
            #x_b += (N-j) * d[(i, N-i)][(j, N-j)]
        #expected_states.append(kl_divergence(normalize([x_a, x_b]),
                                             #normalize([i, N-i])))

    ## Set up plots
    #gs = gridspec.GridSpec(2, 1)
    ##ax1 = pyplot.subplot(gs[0, 0])
    ##ax1.set_title("Transition Probabilities")
    ##ups, downs, _ = two_dim_transitions(edges)
    ##xs = range(0, N+1)
    ##ax1.plot(xs, ups)
    ##ax1.plot(xs, downs)

    #ax2 = pyplot.subplot(gs[1, 0])
    #ax2.set_title("Relative Entropy")
    #divs1 = incentive_process.kl(edges)
    #divs2 = incentive_process.kl(edges, q_d=0)
    #plot_stationary(divs1, ax=ax2)
    #plot_stationary(divs2, ax=ax2)

    #ax3 = pyplot.subplot(gs[2, 0])
    #ax3.set_title("Stationary Distribution")
    #plot_stationary(s, ax=ax3)
    #ax3.set_xlabel("Number of A individuals (i)")

def two_player_example(N=50):
    """
    Tournament Selection for two players.
    """

    m = [[1, 2], [2, 1]]
    #m = [[1, 1], [0, 1]]
    mu = 1. / N
    incentive_func = replicator
    two_dim_transitions_figure(N, m, mu=mu, incentive_func=incentive_func)
    pyplot.show()

if __name__ == '__main__':

    # Two-type example
    two_player_example()
    exit()

    ## Three dimensional examples
    graphical_abstract_figures()
    rps_figures()
    tournament_stationary_3(N=60, mu=1./3)

    ## Four dimensional examples
    four_dim_figures()
