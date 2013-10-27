import math
from matplotlib import pyplot

from incentives import *
import incentive_process
from stationary import approximate_stationary_distribution
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

    
#it's a replicator for the matrix m = [[1,2],[2,1]] with q=-1 (admittedly a little artificial) for the wright-fisher process -- you can see that the lack of a unique interior max is indeed crucial to the proof. Also for m = [[1,2],[5,1]] (with different height peaks).

#However... the theorem is still valid for the incentive process (and the n-fold process). And to make matters worse (better?), I found a non-counterexample for the wright-fisher process:

    #N=50
    #mu = 0.1
    #k = 1
    #q = 1.5
    #mutations = "uniform"
    #incentive = "replicator"
    #m = [[1, 1], [1, 1]]    

## Graphical Abstract ## 

def graphical_abstract_figures(N=60, q=1, beta=0.1):
#def graphical_abstract_figures(N=80, q=1, beta=0.25):
#def graphical_abstract_figures(N=80, q=1, beta=0.5):
#def graphical_abstract_figures(N=80, q=1, beta=0.1, iterations=100):
    #a = 0
    #b = 1
    a = 0
    b = 1
    m = [[a,b,b],[b,a,b],[b,b,a]]
    mu = 1./N
    fitness_landscape = linear_fitness_landscape(m)
    incentive = logit(fitness_landscape, beta=beta, q=q)
    edges = incentive_process.multivariate_transitions(N, incentive, num_types=3, mu=mu)
    d = approximate_stationary_distribution(N, edges, iterations=None)
    heatmap(d, filename="ga_stationary.eps", boundary=True)
    d = incentive_process.kl(N, edges, q_d=0, boundary=True)
    heatmap(d, filename="ga_d_0.eps", boundary=True)
    d = incentive_process.kl(N, edges, q_d=1, boundary=False)
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

#def rock_scissors_paper(a=1, b=1):
    #return [[0,-b,a], [a, 0, -b], [-b, a, 0]]

def k_fold_kl(N=80, k=40, q=1, beta=1., num_types=3):
    a = 1
    b = 1
    m = [[0, -1, 1], [1, 0, -1], [-1, 1, 0]]

    mu = 1./N
    fitness_landscape = linear_fitness_landscape(m)
    incentive = logit(fitness_landscape, beta=beta, q=q)
    edges = incentive_process.multivariate_transitions(N, incentive, num_types=3, mu=mu)
    edge_func = power_transitions(edges, N, k=k)
    from wright_fisher import kl
    d = kl(N, edge_func, q_d=0)
    heatmap(d)
    pyplot.show()
    
def rsp_figures(N=60, q=1, beta=1.):
    m = [[0, -1, 1], [1, 0, -1], [-1, 1, 0]]            
    num_types = len(m[0])
    fitness_landscape = linear_fitness_landscape(m)
    for i, mu in enumerate([1./math.sqrt(N), 1./N, 1./N**2]): 
        # Approximate calculation
        incentive = logit(fitness_landscape, beta=beta, q=q)
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
    mu = 1./N
    # Approximate calculation
    incentive = logit(fitness_landscape, beta=beta, q=q)
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
            full_state = (a,b,0,c)
            temp[state] = d[full_state]
        pyplot.figure()
        heatmap(temp)
        #heatmap(d, filename="rsp_mu_" + str(i) + ".eps", dpi=600)
    pyplot.show()


if __name__ == '__main__':
    graphical_abstract_figures()
    #k_fold_kl()
    #rsp_figured()
    #four_d_figures()
    pass
    
    
    