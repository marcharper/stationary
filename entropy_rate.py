from collections import defaultdict
import math
import pickle

import numpy
from numpy import log, exp, arange

import matplotlib
from matplotlib import pyplot

from math_helpers import kl_divergence_dict, simplex_generator, q_divergence
from incentives import *
import incentive_process
from stationary import approximate_stationary_distribution, exact_stationary_distribution, neutral_stationary
import heatmap
#from three_dim import heatmap
import three_dim

### Global Font config for plots ###

font = {'size': 22}
matplotlib.rc('font', **font)

def entropy_rate(edges, s):
    e = defaultdict(float)
    for a,b,v in edges:
        e[a] -= s[a] * v * log(v)
    return sum(e.values())

def entropy_rate_func(N, edge_func, s):
    e = defaultdict(float)
    for a in simplex_generator(N):
        for b in simplex_generator(N):
            v = edge_func(a,b)
            e[a] -= s[a] * v * log(v)
    return sum(e.values())


def compute_entropy_rate(N=30, n=2, m=None, incentive_func=logit, beta=1., q=1., mu=None, exact=False):
    if not m:
        m = numpy.ones((n,n))
    fitness_landscape = linear_fitness_landscape(m)
    incentive = logit(fitness_landscape, beta=beta, q=1)
    #incentive = replicator(fitness_landscape)
    if not mu:
        #mu = (n-1.)/n * 1./(N+1)
        mu = 1./(N)
    edges = incentive_process.multivariate_transitions(N, incentive, num_types=n, mu=mu)
    if not exact:
        # Approximate Calculation
        s = approximate_stationary_distribution(N, edges, convergence_lim=1e-13)
    else:
        # Exact Calculuation
        edge_dict = dict()
        for e1, e2, v in edges:
            edge_dict[(e1,e2)] = v
        s = exact_stationary_distribution(N, edge_dict, num_players=n)
    e = entropy_rate(edges, s)
    return e
    ##print e, ((2*n-1.)/n * log(n))
    ##d = neutral_stationary(N, alpha, n=n)
    #pyplot.figure()
    ##heatmap(d)
    #domain = range(len(s))
    #pyplot.plot(domain, [s[(i, N-i)] for i in domain])
    #plot_transition_entropy(N, mu, n=2, beta=1., q=1.)
    #pyplot.show()

def mutation_drift_figure_k_data(N=100, n=2, step=0.02, end=2):
    m = numpy.ones((n,n))
    fitness_landscape = linear_fitness_landscape(m)
    #incentive = logit(fitness_landscape, beta=1, q=1)
    incentive = replicator(fitness_landscape)
    ers = []
    domain = arange(step, end, step)
    for k in domain:
        mu = (n-1.)/n * 1./(N+1)**k
        alpha = mu *N / (n-1. - mu * n)
        edges = incentive_process.multivariate_transitions(N, incentive, num_types=n, mu=mu)
        s = neutral_stationary(N, alpha, n)
        e = entropy_rate(edges, s)
        ers.append(e)
    return ers

def figure_1(N=50, kstep=0.02, kend=2, nstart=2, nstop=5, filename="figure_1_data.pickle"):
    """Plots the entropy rate of the neutral landscape versus the scaling parameter k for various n."""
    try:
        data = pickle.load(open(filename, "rb"))
    except:
        print "Computing and Caching data"
        data = []
        for n in range(nstart, nstop+1):
            d = mutation_drift_figure_k_data(N=N, n=n, step=kstep, end=kend)
            data.append((n, d))
            print n, (2*n-1.)/n * log(n), d[-1]
        pickle.dump(data, open(filename, "wb"))
    domain = arange(kstep, kend, kstep)
    for n, d in data:
        pyplot.plot(domain, d, linewidth=2)
        pyplot.axhline((2*n-1.)/n * log(n), linestyle='-', color='black')
    pyplot.xlabel("Scaling Parameter k")
    pyplot.ylabel("Entropy Rate")
    pyplot.show()

def figure_2():
    """Plots the entropy rate as r and beta change."""
    # Beta test
    r = 2
    m = [[r,r],[1,1]]
    domain = arange(0,4,0.01)
    es = []
    for beta in domain:
        e = compute_entropy_rate(m=m, beta=beta, exact=True)
        print beta, e
        es.append(e)
    pyplot.plot(domain, es, linewidth=2)
    
    ## r test
    es = []
    for r in domain:
        m = [[r,r],[1,1]]
        e = compute_entropy_rate(m=m, beta=1., exact=True)
        print r, e
        es.append(e)
    pyplot.plot(domain, es, linewidth=2)
    pyplot.xlabel(r"""Strength of selection $\beta$""" + "\n" +r"""Relative fitness $r$""")
    pyplot.ylabel("Entropy Rate")
    pyplot.show()

def rps_figure(N_range=arange(3,50,1), a_range=arange(0.01,2,0.01), beta=1, filename="rsp_figure_data.pickle"):
#def rps_figure(N_range=arange(40,50,1), a_range=arange(0.01,2,0.01), beta=1, filename="rsp_figure_data.pickle"):
    try:
        es = pickle.load(open(filename, "rb"))
    except:
        es = []
        for N in N_range:
            for a in a_range:
                m = rock_scissors_paper(a=a, b=1)
                e = compute_entropy_rate(N=N, m=m, beta=beta, n=3, exact=False)
                print N, a, e
                es.append((N,a,e))
        pickle.dump(es, open(filename, "wb"))
    xs, ys, cs = heatmap.prepare_heatmap_data(es)
    heatmap.heatmap(xs,ys,cs)
    pyplot.show()

def rps_figure_2(N=20, a_range=arange(-2,2,0.05), b_range=arange(-2,2,0.05), beta=1, filename="rsp_figure_data_2.pickle"):
    try:
        es = pickle.load(open(filename, "rb"))
    except:
        es = []
        for a in a_range:
            for b in a_range:
                m = rock_scissors_paper(a=a, b=b)
                e = compute_entropy_rate(N=N, m=m, beta=beta, n=3, exact=False)
                print a, b, e
                es.append((a,b,e))
        pickle.dump(es, open(filename, "wb"))
    xs, ys, cs = heatmap.prepare_heatmap_data(es)
    heatmap.heatmap(xs,ys,cs)
    pyplot.show()

def compute_bomze_er(N=30):
    """Computes the entropy rate for each of the 48 3x3 matrices in Bomze's classification."""
    mu = 1./N
    for i, m in enumerate(list(three_dim.m_gen())):
        e = compute_entropy_rate(N=N, n=3, m=m, mu=mu, beta=1.)
        print i, e

def traulsen_critical_test():
    b = 0.9
    a = 1
    beta = 6
    m = rock_scissors_paper(a=a, b=b)
    es = []
    for N in range(4,40):
        e = compute_entropy_rate(N=N, m=m, beta=beta, n=3, exact=False)
        print N, e
        es.append(e)
    pyplot.plot(range(4,40), e)
    pyplot.show()

def rsp(N, a, b, mu=None, beta=1.):
    if not mu:
        mu = 1./N
    m = rock_scissors_paper(a=a, b=b)
    e = compute_entropy_rate(N=N, m=m, beta=beta, n=3, exact=False)
    print N, a, b, e

def best_reply_test(N=250, m=[[1,2],[2,1]], beta=10., mu=None, n=2):
    if not mu:
        mu = 1./N
    es = []
    beta_domain = arange(3, 150, 0.5)
    #pyplot.figure()
    for beta in beta_domain:
        fitness_landscape = linear_fitness_landscape(m)
        incentive = logit2(fitness_landscape, beta=beta)
        edges = incentive_process.multivariate_transitions(N, incentive, num_types=n, mu=mu)
        # Exact Calculuation
        edge_dict = dict()
        for e1, e2, v in edges:
            edge_dict[(e1,e2)] = v
        s = exact_stationary_distribution(N, edge_dict, num_players=n)
        #pyplot.plot(range(len(s)), [s[(i, N-i)] for i in range(len(s))])
        e = entropy_rate(edges, s)
        es.append(e)
        print beta, e, (N/2 - 1, N - N/2 +1), s[(N/2 - 1, N - N/2 +1)]
        #from math_helpers import shannon_entropy
    #print shannon_entropy([mu/2., (1.-mu)/2, 1./2])

    pyplot.figure()
    pyplot.plot(beta_domain, es)
    pyplot.show()
    #print N, beta, e

def best_reply_test2(N=250, m=[[1,2],[2,1]], beta=10., mu=None, n=2):
    if not mu:
        mu = 1./N
    #es = []
    
    fitness_landscape = linear_fitness_landscape(m)
    incentive = simple_best_reply(fitness_landscape)
    edges = incentive_process.multivariate_transitions(N, incentive, num_types=n, mu=mu)
    # Exact Calculuation
    edge_dict = dict()
    for e1, e2, v in edges:
        edge_dict[(e1,e2)] = v
    s = exact_stationary_distribution(N, edge_dict, num_players=n)
    #pyplot.plot(range(len(s)), [s[(i, N-i)] for i in range(len(s))])
    e = entropy_rate(edges, s)
    print e
    from math_helpers import shannon_entropy
    print shannon_entropy([mu/2., (1.-mu)/2, 1./2])
    #es.append(e)
    #print beta, e, (N/2 - 1, N - N/2 +1), s[(N/2 - 1, N - N/2 +1)]
    ##from math_helpers import shannon_entropy
    ##print shannon_entropy([mu/2., (1.-mu)/2, 1./2])

    #pyplot.figure()
    #pyplot.plot(beta_domain, es)
    #pyplot.show()
    ##print N, beta, e

def wright_fisher_test(N=40, n=3, mu =0.01, beta=1., m=[[1,1,1],[1,1,1],[1,1,1]], q=1):
    import wright_fisher

    fitness_landscape = linear_fitness_landscape(m)
    incentive = logit(fitness_landscape, beta=beta, q=q)
    edge_func = wright_fisher.multivariate_transitions(N, incentive, mu=mu)
    s = wright_fisher.stationary_distribution(N, edge_func)
    e = entropy_rate_func(N, edge_func, s)
    print e
    print e / (2.*math.log(N))


if __name__ == '__main__':
    #rsp(100,2,1)
    #best_reply_test2(mu=1e-20)
    wright_fisher_test(N=100)
    #figure_1()
    #figure_2()
    #rps_figure()
    #rps_figure_2()
    #traulsen_critical_test()
    exit()


