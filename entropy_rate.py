from collections import defaultdict
import math
import pickle

import numpy
from numpy import log, exp, arange

import matplotlib
from matplotlib import pyplot

from math_helpers import kl_divergence_dict, simplex_generator, q_divergence, edges_to_edge_dict
from incentives import *
import incentive_process

from stationary import compute_stationary, neutral_stationary
import heatmap
#from three_dim import heatmap
import three_dim

### Global Font config for plots ###

font = {'size': 22}
matplotlib.rc('font', **font)

### Compute Entropy Rates ###

def compute_edges(N=30, num_types=2, m=None, incentive_func=logit, beta=1., q=1., mu=None):
    """Calculates the weighted edges of the incentive process."""
    if not m:
        m = numpy.ones((n,n))
    fitness_landscape = linear_fitness_landscape(m)
    incentive = incentive_func(fitness_landscape, beta=beta, q=q)
    if not mu:
        #mu = (n-1.)/n * 1./(N+1)
        mu = 1./(N)
    edges = incentive_process.multivariate_transitions(N, incentive, num_types=num_types, mu=mu)
    return edges

#def compute_stationary(edges, exact=False, convergence_lim=1e-13):
    #if not exact:
        ## Approximate Calculation
        #s = approximate_stationary_distribution(edges, convergence_lim=convergence_lim)
    #else:
        ## Exact Calculuation
        #edge_dict = edges_to_edge_dict(edges)
        #s = exact_stationary_distribution(edge_dict, num_players=n)
    #return s

#def compute_entropy_rate(N=30, n=2, m=None, incentive_func=fermi, beta=1., q=1., mu=None, exact=False, return_stationary=False, convergence_lim=1e-13):
    ##if not m:
        ##m = numpy.ones((n,n))
    ##fitness_landscape = linear_fitness_landscape(m)
    ##incentive = incentive_func(fitness_landscape, beta=beta, q=1)
    ##if not mu:
        ###mu = (n-1.)/n * 1./(N+1)
        ##mu = 1./(N)
    ##edges = incentive_process.multivariate_transitions(N, incentive, num_types=n, mu=mu)
    ##if not exact:
        ### Approximate Calculation
        ##s = approximate_stationary_distribution(N, edges, convergence_lim=convergence_lim)
    ##else:
        ### Exact Calculuation
        ##edge_dict = defaultdict(float)
        ##for e1, e2, v in edges:
            ##edge_dict[(e1,e2)] = v
        ##s = exact_stationary_distribution(N, edge_dict, num_players=n)
    
    #e = entropy_rate(edges, s)
    #if return_stationary:
        #return (e, s)
    #return e

    ###print e, ((2*n-1.)/n * log(n))
    ###d = neutral_stationary(N, alpha, n=n)
    ##pyplot.figure()
    ###heatmap(d)
    ##domain = range(len(s))
    ##pyplot.plot(domain, [s[(i, N-i)] for i in domain])
    ##plot_transition_entropy(N, mu, n=2, beta=1., q=1.)
    ##pyplot.show()

def entropy_rate(edges, stationary):
    """Computes the entropy rate given the edges of the process and the stationary distribution."""
    e = defaultdict(float)
    for a,b,v in edges:
        e[a] -= stationary[a] * v * log(v)
    return sum(e.values())

def entropy_rate_func(N, edge_func, s):
    """Computes entropy rate for a process with a large transition matrix, defined by a transition function (edge_func) rather than a list of weighted edges."""
    e = defaultdict(float)
    for a in simplex_generator(N):
        for b in simplex_generator(N):
            v = edge_func(a,b)
            e[a] -= s[a] * v * log(v)
    return sum(e.values())

###########
# Figures #
###########


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

### Entropy Characterization Plots ###

#compute_entropy_rate(N=30, n=2, m=None, incentive_func=logit, beta=1., q=1., mu=None, exact=False, return_stationary=False)

def dict_max(d):
    k0, v0 = d.items()[0]
    for k,v in d.items():
        if v > v0:
            k0,v0 = k,v
    return k0, v0

## Varying Beta

def ER_figure(N=30):
    """"""
    # Beta test
    #m = [[1,2],[2,1]]
    m = [[1,4],[4,1]]
    domain = arange(0,10,0.1)
    ss = []
    
    plot_data = []
    
    for beta in domain:
        e,s = compute_entropy_rate(N=N, m=m, beta=beta, exact=True, incentive_func=fermi, return_stationary=True)
        #print beta, e
        ss.append(s)
        state, s_max = dict_max(s)

        plot_data.append((s_max, e, e / s_max))

    ax1 = pyplot.subplot2grid((2, 2), (0, 0))
    ax2 = pyplot.subplot2grid((2, 2), (1, 0))
    ax3 = pyplot.subplot2grid((2, 2), (0, 1), rowspan=2)

    #figure, axes = pyplot.subplots(1, 2)
    for i in range(0,2):
        ax1.plot(domain, [x[i] for x in plot_data], linewidth=2)
    i = 2
    ax2.plot(domain, [x[i] for x in plot_data], linewidth=2)
    ax1.set_title("Local Max at %s" % str((N/2, N/2)))
    ax1.set_ylabel("Stationary Maximum, Entropy Rate")
    ax2.set_xlabel("Strength of selection (Beta)")
    ax2.set_ylabel("Random trajectory entropy")
        
        
    for s in ss:
        ax3.plot(range(0, N+1), [s[(i, N-i)] for i in range(0, N+1)])
    ax3.set_title("Stationary Distributions")
    ax3.set_xlabel("Population States")



    #figure, axes = pyplot.subplots(1, 2)
    #for i in range(0,3):
        #axes[0].plot(domain, [x[i] for x in plot_data], linewidth=2)
        
    #for s in ss:
        #axes[1].plot(range(0, N+1), [s[(i, N-i)] for i in range(0, N+1)])

    pyplot.show()
    exit()

def ER_figure_2(N=30, beta=1, iss_state=None):
    """"""
    # Beta test
    #m = [[1,2],[2,1]]
    m = [[0,1,1],[1,0,1],[1,1,0]]
    #m = rock_scissors_paper(a=2, b=1)
    
    
    ax4 = pyplot.subplot2grid((3,2), (0, 1), rowspan=3)
    q=1.
    mu = 1./N
    fitness_landscape = linear_fitness_landscape(m)
    incentive = fermi(fitness_landscape, beta=beta, q=q)
    edges = incentive_process.multivariate_transitions(N, incentive, num_types=3, mu=mu)
    d = approximate_stationary_distribution(N, edges, iterations=None)
    three_dim.heatmap(d, filename="ga_stationary.eps", boundary=False, ax=ax4)
    
    domain = arange(0,1.5,0.1)
    ss = []
    
    plot_data = [[],[]]
    
    iss_states = [(N/2,N/2,0),(N/3,N/3,N/3)]
    
    for beta in domain:
        e,s = compute_entropy_rate(N=N, m=m, n=3, beta=beta, exact=False, incentive_func=fermi, return_stationary=True)
        #print beta, e
        ss.append(s)
        for i, iss_state in enumerate(iss_states):
            state = iss_state
            s_max = s[state]
            plot_data[i].append((s_max, e, e / s_max))
        #else:
            #state, s_max = dict_max(s)
        #print beta, state, s_max

    # Plot Entropy Rate
    ax1 = pyplot.subplot2grid((3,2), (0, 0))
    ax1.plot(domain, [x[1] for x in plot_data[0]], linewidth=2)
    # Plot Stationary Probs
    ax2 = pyplot.subplot2grid((3,2), (1, 0))
    for j in [0,1]:
        ax2.plot(domain, [x[0] for x in plot_data[j]], linewidth=2)
    # Plot entropies
    ax3 = pyplot.subplot2grid((3,2), (2, 0))
    for j in [0,1]:
        ax3.plot(domain, [x[2] for x in plot_data[j]], linewidth=2)


    ax1.set_ylabel("Entropy Rate")
    #ax1.set_title("Blue: %s, Green: %s" % map(str, iss_states))
    ax1.set_title("Blue: (30,30,0), Green: (20,20,20)")

    ax2.set_ylabel("Stationary Local\n Maxima")

    ax3.set_xlabel("Strength of selection (Beta)")
    ax3.set_ylabel("Random trajectory\n entropy")

    pyplot.show()
    exit()

## Varying Population Size N
def ER_figure_N(Ns=None, beta=1, mus=None, m=None, incentive_func=fermi):
    """"""
    import scipy.misc
    if not m:
        m = [[0,1,1],[1,0,1],[1,1,0]]
        #m = rock_scissors_paper(a=1, b=1)
    if not Ns:
        Ns = range(6, 9*6, 6)
        #Ns = range(6, 36, 6)
    n = len(m[0])
    #ax4 = pyplot.subplot2grid((3,2), (0, 1), rowspan=3)
    #fitness_landscape = linear_fitness_landscape(m)
    #incentive = fermi(fitness_landscape, beta=beta, q=1.)
    

    ss = []
    plot_data = [[],[]]
    
    for N in Ns:
        if not mus:
            mu = 1./N
        iss_states = [(N/2,N/2,0),(N/3,N/3,N/3)]
        #edges = incentive_process.multivariate_transitions(N, incentive, num_types=3, mu=mu)
        e,s = compute_entropy_rate(N=N, m=m, n=3, beta=beta, exact=False, incentive_func=incentive_func, return_stationary=True)
        #print beta, e
        ss.append(s)
        norm = scipy.misc.comb(N+n-1, n)
        #norm = math.log(N)
        for i, iss_state in enumerate(iss_states):
            state = iss_state
            s_max = s[state]
            plot_data[i].append((s_max, e, e / (s_max*norm)))

    # Plot Entropy Rate
    domain = Ns
    ax1 = pyplot.subplot2grid((3,2), (0, 0))
    ax1.plot(domain, [x[1] for x in plot_data[0]], linewidth=2)
    # Plot Stationary Probs
    ax2 = pyplot.subplot2grid((3,2), (1, 0))
    for j in [0,1]:
        ax2.plot(domain, [x[0] for x in plot_data[j]], linewidth=2)
    # Plot entropies
    ax3 = pyplot.subplot2grid((3,2), (2, 0))
    for j in [0,1]:
        ax3.plot(domain, [x[2] for x in plot_data[j]], linewidth=2)


    ax1.set_ylabel("Entropy Rate")
    #ax1.set_title("Blue: %s, Green: %s" % map(str, iss_states))
    ax1.set_title("Blue: (N/2,N/2,0), Green: (N/3,N/3,N/3)")

    ax2.set_ylabel("Stationary Local\n Maxima")

    ax3.set_xlabel("Population Size (N)")
    ax3.set_ylabel("Random trajectory\n entropy")

    pyplot.show()


## Varying Mutation Rates

def ER_figure_mu(N=30, beta=1., mus=None, m=None, incentive_func=fermi):
    """"""
    if not m:
        m = [[0,1,1],[1,0,1],[1,1,0]]
        #m = rock_scissors_paper(a=2, b=1)
    if not mus:
        mus = map(lambda x: math.exp(-math.log(10)*x), range(2,5,1))
        mus = [x/float(N**2) for x in range(1, N+1)] 
        
        #mus.extend((1./N**2, 1./(2*N), 1./N))
        mus.sort()

    #ax4 = pyplot.subplot2grid((3,2), (0, 1), rowspan=3)
    #fitness_landscape = linear_fitness_landscape(m)
    #incentive = fermi(fitness_landscape, beta=beta, q=1.)
    

    ss = []
    plot_data = [[],[]]
    
    for mu in mus:
        print mu
        iss_states = [(N/2,N/2,0),(N/3,N/3,N/3)]
        #edges = incentive_process.multivariate_transitions(N, incentive, num_types=3, mu=mu)
        e,s = compute_entropy_rate(N=N, m=m, n=3, beta=beta, exact=False, mu=mu, incentive_func=incentive_func, return_stationary=True, convergence_lim=1e-8)
        ss.append(s)
        for i, iss_state in enumerate(iss_states):
            state = iss_state
            s_max = s[state]
            plot_data[i].append((s_max, e, e / s_max))

    # Plot Entropy Rate
    domain = mus
    ax1 = pyplot.subplot2grid((3,2), (0, 0))
    ax1.plot(domain, [x[1] for x in plot_data[0]], linewidth=2)
    # Plot Stationary Probs
    ax2 = pyplot.subplot2grid((3,2), (1, 0))
    for j in [0,1]:
        ax2.plot(domain, [x[0] for x in plot_data[j]], linewidth=2)
    # Plot entropies
    ax3 = pyplot.subplot2grid((3,2), (2, 0))
    for j in [0,1]:
        ax3.plot(domain, [x[2] for x in plot_data[j]], linewidth=2)


    ax1.set_ylabel("Entropy Rate")
    #ax1.set_title("Blue: %s, Green: %s" % map(str, iss_states))
    ax1.set_title("Blue: (N/2,N/2,0), Green: (N/3,N/3,N/3)")

    ax2.set_ylabel("Stationary Local\n Maxima")

    ax3.set_xlabel("Mutation rate (mu)")
    ax3.set_ylabel("Random trajectory\n entropy")

    pyplot.show()



if __name__ == '__main__':
    #rsp(100,2,1)
    #best_reply_test2(mu=1e-20)
    #wright_fisher_test(N=100)
    #figure_1()
    #figure_2()
    #rps_figure()
    #rps_figure_2()
    #traulsen_critical_test()
    #ER_figure()
    #ER_figure_2(N=60, beta=0.1)
    #ER_figure_2(N=60, beta=3.)
    #ER_figure_N(Ns=None, beta=1, mus=None, m=None)
    ER_figure_mu(N=30, beta=1, mus=None, m=None, incentive_func=fermi)
    
    exit()


