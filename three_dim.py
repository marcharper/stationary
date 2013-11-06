"""Compute E(x), KL, plot heatmap; plot stationary."""
from collections import defaultdict
from itertools import izip
import math
import multiprocessing
import os
from itertools import izip

import numpy
from numpy import array, log, exp
from matplotlib import pyplot

import ternary
#from mpsim.stationary import *
from incentives import *
from math_helpers import kl_divergence, kl_divergence_dict, one_step_generator

import incentive_process
import wright_fisher
from stationary import approximate_stationary_distribution

numpy.seterr(all="print")

# http://www.statslab.cam.ac.uk/~frank/BOOKS/book/ch1.pdf

##############
## Plotting ##
##############

def heatmap(d, filename=None, boundary=False, dpi=200):
    d2 = dict()
    for (i,j,k), v in d.items():
        N = i+j+k
        # Skip the boundary states
        if not boundary:
            if i*j*k == 0:
                continue
        d2[(k,j)] = v
    ternary.heatmap(d2, N)
    if boundary:
        pyplot.xlim(0,N+1)
    else:
        pyplot.xlim(0,N)
    pyplot.ylim(0, (N*math.sqrt(3)/2) + 2)
    if filename:
        pyplot.savefig(filename, dpi=dpi)

def m_gen(filename="bomze.txt"):
    handle = open(filename)
    for line in handle:
        a,b,c,d,e,f,g,h,i = map(float, line.split())
        yield [[a,b,c],[d,e,f],[g,h,i]]

def bomze_plots(N=40, m=None, i=0, directory="plots", beta=1., q=1., q_ds=None, mu=0.001, iterations=100, dpi=200, process="incentive", boundary=True):
    #if not os.path.isdir(directory):
        #os.mkdir(directory)
    print process, i
    fitness_landscape = linear_fitness_landscape(m)
    if process == "incentive":
        incentive = fermi(fitness_landscape, beta=beta, q=q)
        edges = incentive_process.multivariate_transitions(N, incentive, num_types=3, mu=mu)
        d = approximate_stationary_distribution(N, edges, iterations=iterations)
    elif process == "wright_fisher":
        incentive = fermi(fitness_landscape, beta=beta, q=q)
        edge_func = wright_fisher.multivariate_transitions(N, incentive, mu=mu)
        d = wright_fisher.stationary_distribution(N, edge_func, iterations=iterations)
    print process, i, "stationary heatmap"
    filename = os.path.join(directory, "%s_%s_stationary.eps" % (i, N))
    heatmap(d, filename=filename, boundary=boundary)    
    if not q_ds:
        q_ds = [1.]
    for q_d in q_ds:
        if process == "incentive":
            d = incentive_process.kl(N, edges, q_d=q_d)
        elif process == "wright_fisher":
            d = wright_fisher.kl(N, edge_func, q_d=q_d)
        print process, i, "heatmap", q_d
        filename = os.path.join(directory, "%s_%s_%s_kl.eps"  % (i, N, q_d))
        heatmap(d, filename=filename, boundary=boundary)    

###############################
### Multiprocessing support ###
###############################

def constant_generator(x):
    while True:
        yield x

def batch_plots(args):
    bomze_plots(*args)

def params_gen(constant_parameters, variable_parameters):
    """Manage parameters for multiprocessing. Functional parameters cannot be pickled, hence this workaround."""    
    parameters = []

    for p, default in [("N", 20), ('m', None), ('i',0), ("directory", "incentive_plots"), ("beta", 1),  ("q", 1), ("q_ds", 1), ("mu", 0.01), ("iterations", 100), ("dpi", 200), ("process", "incentive")]:
        try:
            value = variable_parameters[p]
        except KeyError:
            try:
                value = constant_generator(constant_parameters[p])
            except KeyError:
                value = constant_generator(default)
        parameters.append(value)
    return izip(*parameters)
   
def run_batches(constant_parameters, variable_parameters, num_processes=8, func=None):
    """Runs calculations on multiple processing cores."""
    if not num_processes:
        num_processes = multiprocessing.cpu_count()
    if not func:
        func = batch_plots
    params = params_gen(constant_parameters, variable_parameters)
    pool = multiprocessing.Pool(processes=num_processes)
    try:
        results = pool.map(func, params)
        pool.close()
        pool.join()
    except KeyboardInterrupt:
        print 'Control-C received. Exiting.'
        pool.terminate()
        exit()

def run_bomze_batches(process="incentive", N=20, directory="plots", mu=0.001, iterations=100, beta=1., q=1., q_ds=None, num_processes=8):
    ms = list(m_gen())
    variable_parameters = dict(m=ms, i=range(len(ms)))
    constant_parameters = dict(N=N, directory=directory, beta=beta, q=q, mu=mu, iterations=iterations, dpi=200, process=process, q_ds=q_ds)
    run_batches(constant_parameters, variable_parameters, num_processes=num_processes)

def find_local_extrema(d, dim=2, extremum="min"):
    states = []
    for state, value in d.items():
        is_extremum = True
        for one_step in one_step_generator(dim):
            adj = tuple(numpy.array(state) + numpy.array(one_step))
            try:
                v2 = d[adj]
            except KeyError:
                continue
            if extremum == "min":
                if value > v2:
                    is_extremum = False
                    break
            if extremum == "max":
                if value < v2:
                    is_extremum = False
                    break
        if is_extremum:
            states.append(state)
    return states

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

from collections import defaultdict
def probability_difference(N, edges, boundary=False):
    """Computes the KL-div of the expected state with the state, for all states."""
    #dist = q_divergence(q_d)
    e = defaultdict(float)
    for x, y, w in edges:
        e[y] += w
        e[x] -= w
    for k,v in e.items():
        e[k] = abs(v)
    return e

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

    pyplot.show()

if __name__ == '__main__':
    #N=120
    #mu = 1./math.sqrt(N)
    ##mu = 1./N
    #m = rock_scissors_paper(a=1, b=1)
    #probability_neutral_check(N=N, m=m, beta=1., q=1., mu=mu)
    #exit()

    
    #N=150
    #m = rock_scissors_paper(a=1, b=1)
    #probability_neutral_check(N=N, m=m, beta=1., q=1., mu=1./N)
    #exit()
    
    ##m = [[1,1,0],[0,1,1],[1,0,1]]
    #m = rock_scissors_paper(a=1, b=-2)
    #bomze_plots(N=20, m=m, i=0, directory="wf_d_1", beta=1., q=1., mu=0.01, q_d=1, iterations=None, dpi=200, process="wright_fisher")
    #exit()
    ##bomze_plots(N=20, m=[[1,1,0],[0,1,1],[1,0,1]], i=0, directory="plots", beta=1., q=1., mu=0.001, iterations=10000, dpi=200, process="wright_fisher")
    #exit()
    
    ##exp_check()
    ##exit()
    
    #bomze_plots(N=20, m=[[1,1,1],[1,2,1],[3,1,3]], i=3, directory="plots", beta=1., q=1., mu=0.01, iterations=100, dpi=200, process="incentive", q_ds=(0.,1.))
    #exit()

    q_ds = (0, 0.5, 1)
    #for process in ("incentive", "wright_fisher"):
    #for process in ("wright_fisher",):
    for process in ("incentive",):
        #for N in (10, 20, 30, 40):
        for N in [80]:
            mu = (3./2.)*1./N
            directory="bomze_%s" % process
            if not os.path.isdir(directory):
                os.mkdir(directory)
            run_bomze_batches(process=process, N=N, q_ds=q_ds, directory=directory, mu=mu, iterations=None, beta=1., num_processes=4)

    exit()
    
    m=[[0,1,1,1],[1,0,1,1],[1,1,0,1], [1,1,1,0]]
    local_min_check(m=m)
    exit()
    
    #m = [[1,1,0],[0,1,1],[1,0,1]]
    m = rock_scissors_paper(a=1, b=-2)
    bomze_plots(N=50, m=m, i=0, directory="plots", beta=1., q=1., mu=0.01, iterations=10000, dpi=200, process="incentive")
    #bomze_plots(N=20, m=[[1,1,0],[0,1,1],[1,0,1]], i=0, directory="plots", beta=1., q=1., mu=0.001, iterations=10000, dpi=200, process="wright_fisher")
    exit()
    
    N = 50
    #mu = 1./(10*N)
    mu = 0.01
    run_bomze_batches(process="incentive", N=N, directory="incentive_plots", mu=mu, iterations=100, beta=1., q=1.)
    #run_bomze_batches(process="wf", N=N, directory="wf_plots", mu=mu, iterations=100)
    exit()
    
    