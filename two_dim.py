from collections import defaultdict
import os

from matplotlib import pyplot
import numpy
from numpy import cumprod, cumsum

#from entropy_rate import compute_entropy_rate

from incentives import *
import incentive_process

from math_helpers import kl_divergence, normalize, dot_product
from stationary import exact_stationary_distribution

# Font config for plots
import matplotlib
font = {'size': 22}
matplotlib.rc('font', **font)

# For Incentive process
def transitions_figure(N, m, mu=0.01, k=1., mutations='uniform', process="moran", incentive_func=replicator, q=1., eta=None):
    """Plot transition entropies and stationary distributions."""
    n = len(m[0])
    fitness_landscape = linear_fitness_landscape(m)
    incentive = incentive_func(fitness_landscape)
    if not mu:
        #mu = (n-1.)/n * 1./(N+1)
        mu = 1./(N)
    edges = incentive_process.multivariate_transitions(N, incentive, num_types=n, mu=mu)

    # Exact Calculuation
    edge_dict = defaultdict(float)
    for e1, e2, v in edges:
        edge_dict[(e1,e2)] = v
    s = exact_stationary_distribution(N, edge_dict, num_players=n)

    d = edge_dict

    ups = []
    downs = []
    for i in range(0, N+1):
        if i != N:
            #ups.append(d[(i,i+1)])
            ups.append(d[((i,N-i), (i+1, N-i-1))])
        else:
            ups.append(0)
        if i != 0:
            #downs.append(d[(i, i-1)])
            downs.append(d[((i, N-i), (i-1, N-i+1))])
        else:
            downs.append(0)
    evens = []
    for i in range(0, N+1):
        evens.append(1.-ups[i]-downs[i])

    pyplot.subplot(311)
    xs = range(0, N+1)
    pyplot.plot(xs, ups)
    pyplot.plot(xs, downs)
    pyplot.title("Transition Probabilities")

    ax = pyplot.subplot(312)
    ee, vs, ss = exp_var_kl(ups, downs)

    pyplot.plot(xs[1:N], ss[1:N])
    pyplot.xlim(0,N)
    pyplot.title("Relative Entropy")
    ax.ticklabel_format(style='sci', scilimits=(0,0), axis='y')

    pyplot.subplot(313)
    
    s_to_plot = [0]*(N+1)
    for (i,j), value in s.items():
        s_to_plot[i] = value
    pyplot.plot(xs, s_to_plot)
    pyplot.title("Stationary Distribution")
    pyplot.xlabel("Number of A individuals (i)")

# For Wright-Fisher and k-fold incentive
def transitions_figure_N(N, m, mu=0.01, k=1., mutations='uniform', process="wright-fisher", incentive="replicator", q=1., eta=None):
    """Plot transition entropies and stationary distributions."""
    (e, s, d) = compute_entropy_rate(N, m=m, mutations=mutations, mu_ab=mu, mu_ba=k*mu, verbose=False, report=True, q=q, process=process, eta=eta)
    pyplot.subplot(311)

    #from numpy import linalg as LA
    #print list(sorted(LA.eigvals(d)))

    xs = range(0, N+1)
    ups = [d[i][i+1] for i in range(0, N)]
    ups.append(0)
    downs = [0]
    downs.extend([d[i][i-1] for i in range(1, N+1)])

    pyplot.plot(xs, ups)
    pyplot.plot(xs, downs)
    pyplot.title("Transition Probabilities")


    pyplot.subplot(312)

    expected_states = []    
    for i in xs:
        x_a = 0.
        x_b = 0
        for j in range(0, N+1):
            x_a += j * d[i][j]
            x_b += (N-j) * d[i][j]
        expected_states.append(kl_divergence(normalize([x_a, x_b]), normalize([i, N-i])))
        #expected_states.append(kl_divergence( normalize([i, N-i]), normalize([x_a, x_b])))

    argmin = 1+ numpy.argmin(expected_states[1:N])
    print "ekl min at i =", argmin

    pyplot.plot(xs[1:N], expected_states[1:N])
    pyplot.xlim(0,N)
    pyplot.title("Relative Entropy")
        #pyplot.ylim(0, (N*math.sqrt(3)/2) + 2)

    pyplot.subplot(313)

    print "stationary max:", numpy.argmax(s)
    #pyplot.plot(range(1, N), s[1:N])
    pyplot.plot(xs, s)
    pyplot.title("Stationary Distribution")
    pyplot.xlabel("Number of A individuals (i)")

if __name__ == '__main__':

    N = 50
    mu = 1./25
    m = [[1,0],[0,1]]    
    incentive_func = replicator

    transitions_figure(N, m, mu=mu, incentive_func=incentive_func, process="moran")
    pyplot.show()
    exit()

    # Tournament Selection    
    m = [[1,1],[0,1]]
    N = 200
    #k = 4./5
    #k = 1.
    k = 1./10
    mu = 1./N**k
    #mu = 0.25
    print mu

    incentive_func = replicator

    transitions_figure(N, m, mu=mu, incentive_func=incentive_func, process="moran")
    pyplot.show()
    exit()


    
    
    
    # Paper "figure_1.eps"
    N= 100
    mu = 1./1000
    k = 1
    q = 1

    mutations = "uniform"
    incentive_func = replicator
    m = [[1, 2], [3, 1]]
    #(e, s, d) = compute_entropy_rate(N, m=m, mutations=mutations, mu_ab=mu, mu_ba=k*mu, verbose=False, report=True, q=q, process="moran")
    
    transitions_figure(N, m, mu=mu, q=q, k=k, mutations=mutations, incentive_func=incentive_func, process="moran")
    pyplot.show()
    exit()
    
    ## basic examples

    N = 400
    #m = [[1,2],[2,1]]
    #m = [[1,1],[1,1]]
    ###m = [[2,2],[1,1]]
    #m = [[1,2],[5,1]]
    m = [[20,1],[7,10]]

    ##transitions_figure(N, m, mu=0.001, q=1., k=1, incentive="replicator")
    #transitions_figure_N(N, m, mu=0.001, q=1., k=1, incentive="replicator", process="n-fold-moran")

    ###transitions_figure(N, m, mu=0.01, q=2., k=10, incentive="replicator")
    ###transitions_figure_N(N, m, mu=0.01, q=2., k=10, incentive="replicator", process="n-fold-moran")
    
    transitions_figure_N(N, m, mu=0.1, q=2., k=2., incentive="replicator", process="wright-fisher")
    transitions_figure_N(N, m, mu=0.01, q=0.9, k=2., incentive="replicator", process="wright-fisher")
    transitions_figure_N(N, m, mu=0.01, q=0., k=1., incentive="replicator", process="wright-fisher")
    