import math
import os
import matplotlib
from matplotlib import pyplot

from incentives import *
import incentive_process
from stationary import approximate_stationary_distribution, edges_to_edge_dict
from three_dim import heatmap

from math_helpers import simplex_generator, q_divergence

#def figure_2(N=100, m=[[1,2],[3,1]], q=1):
    #"""Demonstration of theorem 1."""
    #transitions_figure(N, m, mu=0.001, q=q, k=1, incentive="replicator", process="moran")
    #pyplot.show()
    
#def figure_3(N=100, m=[[1,2],[2,1]], q=-1, mu=0.1):   
    #transitions_figure_N(N, m, mu=mu, q=q, k=1, incentive="replicator", process="wright-fisher")
    #pyplot.show()

#def figure_4(N=100, m=[[1,1],[1,1]], q=1.5, mu=0.1):   
    #transitions_figure_N(N, m, mu=mu, q=q, k=1, incentive="replicator", process="wright-fisher")
    #pyplot.show()

## Graphical Abstract ## 

def graphical_abstract_figures(N=60, q=1, beta=0.1):
    a = 0
    b = 1
    m = [[a,b,b],[b,a,b],[b,b,a]]
    mu = (3./2)*1./N
    fitness_landscape = linear_fitness_landscape(m)
    incentive = fermi(fitness_landscape, beta=beta, q=q)
    edges = incentive_process.multivariate_transitions(N, incentive, num_types=3, mu=mu)
    d = approximate_stationary_distribution(N, edges, iterations=None)
    heatmap(d, filename="ga_stationary.eps", boundary=False)
    d = incentive_process.kl(N, edges, q_d=0)
    heatmap(d, filename="ga_d_0.eps", boundary=False)
    d = incentive_process.kl(N, edges, q_d=1)
    heatmap(d, filename="ga_d_1.eps", boundary=False)

def power_transitions(edges, N, k=20):
    # enumerate states
    enum = dict()
    inv = dict()
    for i, current_state in enumerate(simplex_generator(N, 2)):
        enum[current_state] = i
        inv[i] = current_state
    M = numpy.zeros(shape=(len(enum), len(enum)))
    for current_state, next_state, v in edges:
        M[enum[current_state]][enum[next_state]] = v
    from numpy import linalg
    M_k = linalg.matrix_power(M, k)
    def edge_func(current_state, next_state):
        return M_k[enum[current_state]][enum[next_state]]
    return edge_func

def k_fold_kl(N=80, k=40, q=1, beta=1., num_types=3, q_d=1):
    a = 1
    b = 1
    m = [[0, -1, 1], [1, 0, -1], [-1, 1, 0]]

    mu = 1./N
    fitness_landscape = linear_fitness_landscape(m)
    incentive = fermi(fitness_landscape, beta=beta, q=q)
    edges = incentive_process.multivariate_transitions(N, incentive, num_types=3, mu=mu)
    edge_func = power_transitions(edges, N, k=k)
    from wright_fisher import kl
    d = kl(N, edge_func, q_d=q_d)
    heatmap(d)
    pyplot.show()
    
    
def rsp_figures(N=60, q=1, beta=1.):
    m = [[0, -1, 1], [1, 0, -1], [-1, 1, 0]]            
    num_types = len(m[0])
    fitness_landscape = linear_fitness_landscape(m)
    for i, mu in enumerate([1./math.sqrt(N), 1./N, 1./N**(3./2)]): 
        # Approximate calculation
        mu = 3/2. * mu
        incentive = fermi(fitness_landscape, beta=beta, q=q)
        edges = incentive_process.multivariate_transitions(N, incentive, num_types=num_types, mu=mu)
        d = approximate_stationary_distribution(N, edges)
        heatmap(d, filename="rsp_mu_" + str(i) + ".eps", dpi=600)


def four_d_figures(N=30, beta=1., q=1.):
    #slice_val = N / 8
    #slice_val = N / 4
    #m = [[0, 1, 1, 1], [1, 0, 1, 1], [1, 1, 0, 1], [1,1,1,0]]
    #a,b,c = 1,2,3
    #m = [[0, a, b, c], [c, 0, a, b], [b, c, 0, a], [a,b,b,0]]
    #m = [[0, -1./2, 1./2, -1./4], [1./2, 0, 1, -1], [-1./2, -1, 0, 1], [1./4,1,-1,0]]
    
    m = [[0, 1, 1, 1], [1, 0, 1, 1], [1, 1, 0, 1], [0,0,0,1]]
    #m = [[0, 1, 1, 1], [1, 0, 1, 1], [1, 1, 0, 1], [1,1,1,0]]
    
    num_types = len(m[0])
    fitness_landscape = linear_fitness_landscape(m)
    mu = 4./3*1./N
    # Approximate calculation
    incentive = fermi(fitness_landscape, beta=beta, q=q)
    edges = incentive_process.multivariate_transitions(N, incentive, num_types=num_types, mu=mu)
    
    d1 = incentive_process.kl(N, edges, q_d=0, boundary=True)
    d2 = approximate_stationary_distribution(N, edges)
        
    for d in [d1, d2]:    
        temp = dict()
        slice_val = 0
        for state in simplex_generator(N - slice_val, 2):
            a,b,c = state
            #full_state = (a,b,c, slice_val)
            #full_state = (0, a,b,c)
            full_state = (a, b,0,c)
            temp[state] = d[full_state]
        pyplot.figure()
        heatmap(temp)    
    pyplot.show()
    for d in [d1, d2]:    
        temp = dict()
        for state in simplex_generator(N - slice_val, 2):
            a,b,c = state
            #full_state = (a,b,c, slice_val)
            #full_state = (a,b,c,0)
            full_state = (a,b,c,0)
            temp[state] = d[full_state]
        pyplot.figure()
        heatmap(temp)
        #heatmap(d, filename="rsp_mu_" + str(i) + ".eps", dpi=600)
    pyplot.show()

def bomze_figures(N=80, indices=[2,20,47], beta=1, process="incentive", directory=None):
    if not directory:
        directory = "bomze_paper_figures_%s" % process
        if not os.path.isdir(directory):
            os.mkdir(directory)
    from three_dim import m_gen, bomze_plots
    for i, m in enumerate(m_gen()):
        if i not in indices:
            continue
        print i
        mu = 3./2 * 1./N
        #mu = (1./2)*1./N
        bomze_plots(N=N, m=m, mu=mu, i=i, directory=directory, beta=beta, q_ds=(0., 0.5, 1.0), process=process)

def rps_stationary(N=50, beta=1):
    m = [[0, -1, 1], [1, 0, -1], [-1, 1, 0]]            
    num_types = len(m[0])
    fitness_landscape = linear_fitness_landscape(m)
    mu = 3./2 * 1./N
    #mu = 1./N
    incentive = fermi(fitness_landscape, beta=beta, q=1)
    edges = incentive_process.multivariate_transitions(N, incentive, num_types=num_types, mu=mu)
    d = approximate_stationary_distribution(N, edges)
    heatmap(d, boundary=True)
    pyplot.show()

def stationary_test(N=80, boundary=True):
    m = rock_scissors_paper(a=4,b=1)
    #m = [[0,1,1], [1,0,1], [1,1,0]]
    num_types = len(m[0])
    fitness_landscape = linear_fitness_landscape(m)
    mu = 3./2 * 1./N
    #mu = 1./N
    incentive = dash_incentive(fitness_landscape)
    edges = incentive_process.multivariate_transitions(N, incentive, num_types=num_types, mu=mu)
    d = approximate_stationary_distribution(N, edges)
    pyplot.figure()
    heatmap(d, boundary=boundary)
    pyplot.figure()
    d = incentive_process.kl(N, edges, q_d=1)
    heatmap(d, boundary=boundary)
    pyplot.show()

def rsp_N_test(N=60, q=1, beta=1., convergence_lim=1e-9, mu=None):
    m = [[0, -1, 1], [1, 0, -1], [-1, 1, 0]]
    num_types = len(m[0])
    fitness_landscape = linear_fitness_landscape(m)
    if not mu:
        mu = 3./2*1./N
    incentive = fermi(fitness_landscape, beta=beta, q=q)
    edges = incentive_process.multivariate_transitions(N, incentive, num_types=num_types, mu=mu)
    #edge_dict = edges_to_edge_dict(edges)
    d = approximate_stationary_distribution(edges, convergence_lim=convergence_lim)
    
    #grid_spec = matplotlib.gridspec.GridSpec(1, 2)
    grid_spec = matplotlib.gridspec.GridSpec(1, 1)
    grid_spec.update(hspace=0.5)
    ax1 = pyplot.subplot(grid_spec[0, 0])
    heatmap(d, ax=ax1, scientific=True)
    #pyplot.show()
    
    #ax2 = pyplot.subplot(grid_spec[0, 1])
    #d = incentive_process.kl(N, edges)
    #heatmap(d, ax=ax2, boundary=False)
    pyplot.savefig(str(N) + ".eps", dpi=500)
    #pyplot.show()
    import cPickle as pickle
    with open(str(N) + "_stationary.pickle", "wb") as handle:
        pickle.dump(d, handle)

def rsp_N_test2(N=60, q=1, beta=1., mu=None, convergence_lim=1e-9):
    m = [[0, -1, 1], [1, 0, -1], [-1, 1, 0]]
    num_types = len(m[0])
    fitness_landscape = linear_fitness_landscape(m)
    # Approximate calculation
    if not mu:
        mu = 3/2. * 1./N
    incentive = fermi(fitness_landscape, beta=beta, q=q)
    edges = incentive_process.multivariate_transitions(N, incentive, num_types=num_types, mu=mu)
    #edge_dict = edges_to_edge_dict(edges)
    #d = approximate_stationary_distribution(edges, convergence_lim=convergence_lim)
    
    #grid_spec = matplotlib.gridspec.GridSpec(1, 2)
    grid_spec = matplotlib.gridspec.GridSpec(1, 1)
    grid_spec.update(hspace=0.5)
    ax1 = pyplot.subplot(grid_spec[0, 0])
    #heatmap(d, ax=ax1, scientific=True)
    #pyplot.show()
    
    #ax2 = pyplot.subplot(grid_spec[0, 1])
    d = incentive_process.kl(N, edges)
    heatmap(d, ax=ax1, boundary=False)
    pyplot.savefig(str(N) + "_kl.eps", dpi=500)
    #pyplot.show()

#def rsp_iss_test(N=60, q=1, beta=1., mu=None, d=2):
    #from math_helpers import simplex_generator, one_step_indicies_generator, kl_divergence
    #from incentive_process import is_valid_state
    #m = [[0, -1, 1], [1, 0, -1], [-1, 1, 0]]
    #num_types = len(m[0])
    #fitness_landscape = linear_fitness_landscape(m)
    #if not mu:
        #mu = 1./N
    #mu = 3/2. * mu
    #incentive = fermi(fitness_landscape, beta=beta, q=q)
    #edges = incentive_process.multivariate_transitions(N, incentive, num_types=num_types, mu=mu)
    #edge_dict = edges_to_edge_dict(edges)
    
    #one_step_indicies = list(one_step_indicies_generator(d))
    #lower, upper = 0, N
    #k = dict()
    
    #for state in simplex_generator(N, d=d):
        #p = []
        #q = []
        #for plus_index, minus_index in one_step_indicies:
            #target_state = list(state)
            #target_state[plus_index] += 1
            #target_state[minus_index] -= 1
            #target_state = tuple(target_state)
            ## Is this a valid state? I.E. Are we on or near the boundary?
            #if not is_valid_state(N, target_state, lower, upper):
                #continue
            #p.append(edge_dict[(state, target_state)])
            #q.append(state[plus_index])
        #p.append(edge_dict[(state, state)])
        #q.append(state[plus_index])
        #print state, p, q
        #k[state] = kl_divergence(normalize(p), normalize(q))
        
    #grid_spec = matplotlib.gridspec.GridSpec(1, 1)
    #grid_spec.update(hspace=0.5)
    #ax1 = pyplot.subplot(grid_spec[0, 0])
    #heatmap(k, ax=ax1, boundary=True)
    #pyplot.show()

        


if __name__ == '__main__':
    N = 120
    mu = 1./N * 3./2
    rsp_N_test(N=N, beta=1., mu=mu, convergence_lim=1e-14)
    #stationary_test()
    #phase_portrait()
    #graphical_abstract_figures(beta=1.)
    #k_fold_kl()
    #rsp_figures()
    #rps_stationary(N=80)
    #four_d_figures(N=40)
    #bomze_figures(N=80, indices=[2,20,47], process="incentive")
    #bomze_figures(N=40, process="incentive")
    
    pass
    
    
    