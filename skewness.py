from collections import defaultdict
import math
import os
import pickle

import numpy
from numpy import log, exp, arange

import matplotlib
from matplotlib import pyplot

from math_helpers import kl_divergence_dict, simplex_generator, q_divergence
from incentives import *
from incentive_process import compute_edges
from stationary import compute_stationary, neutral_stationary, entropy_rate
import heatmap
#from three_dim import heatmap
import three_dim

from mpsim.stationary import Cache, Graph, stationary_distribution_generator

import subprocess
from two_dim import ensure_directory, ensure_digits

### Global Font config for plots ###

matplotlib.rc('font', size=28)
matplotlib.rc('font', family='serif')
matplotlib.rc('text', usetex=True)


def rsp(N, a, b, mu=None, beta=1.):
    if not mu:
        mu = 1./N
    m = rock_scissors_paper(a=a, b=b)
    e = compute_entropy_rate(N=N, m=m, beta=beta, n=3, exact=False)
    print N, a, b, e

def skewness_heatmap(N=30, num_types=2, m=[[1,2],[2,1]], beta=0.1, exact=True):
    """Heatmap of skewness for two dim game for every pair of starting and ending state."""

    edges = compute_edges(N=N, m=m, beta=beta, incentive_func=fermi, num_types=num_types)
    s = compute_stationary(edges, exact=True)

    data = []
    for i in range(0, N+1):
        for j in range(0, N+1):
            skew = - math.log(s[(i, N-i)]) + math.log(s[(j, N-j)])
            data.append((i,j,skew))
    heatmap.main(data, xfunc=int, yfunc=int, rounding=False)
    pyplot.xlim(0, N)
    pyplot.ylim(0, N)
    pyplot.xlabel("Initial State $(i, N-i)$")
    pyplot.ylabel("Final State $(j, N-j)$")
    pyplot.title("Cumulative Skewness")


def invert_enumeration(cache, ranks):
    d = dict()
    for m, r in enumerate(ranks):
        state = cache.inv_enum[m]
        d[(state)] = r
    return d

def transition_matrix_power(edges, initial_state=None, power=20, yield_all=True):
    g = Graph()
    g.add_edges(edges)
    #g.normalize_weights()
    cache = Cache(g)
    N = sum(edges[0][0])
    initial_state_ = [0]*(N+1)
    if not initial_state:
        initial_state_[cache.enum[(1,N-1)]] = 1
    else:
        initial_state_[cache.enum[initial_state]] = 1

    gen = stationary_distribution_generator(cache, initial_state=initial_state_)
    for i, ranks in enumerate(gen):
        if not yield_all:
            if i == power:
                break
        else:
            yield invert_enumeration(cache, ranks)
    yield invert_enumeration(cache, ranks)

def self_info_plot(N=30, m=[[1,2],[2,1]], beta=0.1, power=20): 
    #s = compute_stationary(N=N, m=m, n=n, beta=beta, exact=exact, incentive_func=fermi, return_stationary=True)

    edges = compute_edges(N=N, m=m, beta=beta, mu=mu)
    final_dist = list(transition_matrix_power(edges, power=power, yield_all=False))[-1]

    data = []
    for i in range(0, N+1):
        try:
            data.append(- math.log(final_dist[(i, N-i)]))
        except ValueError:
            data.append(0)
    pyplot.plot(range(0, N+1), data)
    pyplot.show()

def self_info_movie(N=30, n=2, m=[[1,2],[2,1]], beta=0.1, max_power=3000, directory="self_info_frames"): 
    edges = compute_edges(N=N, m=m, beta=beta, mu=mu)
    final_dist = transition_matrix_power(edges, power=power, generator=True)

    for j, final_dist in enumerate(final_dist_gen):
        if j == max_power:
            break
        data = []
        for i in range(0, N+1):
            try:
                data.append(- math.log(final_dist[(i, N-i)]))
            except ValueError:
                data.append(0)
        pyplot.clf()
        pyplot.plot(range(0, N+1), data)
        pyplot.title("Iteration %s" % str(j))
        pyplot.ylim(2, 7)
        digits = len(str(max_power))
        filename = os.path.join(directory, ensure_digits(digits, str(j)) + ".png")
        print j, "saving", filename
        pyplot.savefig(filename, dpi=250)
    
    framerate = 20
    movie_filename = "movie.mp4"
    # avconv -r 20 -i "%04d.png" -b:v 1000k test.mp4
    subprocess.call(["avconv", "-r", '\"' + str(framerate), "-i", "%0" + str(digits) + "d.png" + '\"', "-b:v", "1000k", movie_filename] )

def self_info_heatmap(N=30, m=[[1,2],[2,1]], beta=0.1, power=20, mu=None): 
    edges = compute_edges(N=N, m=m, beta=beta, mu=mu)
    data = []
    for j in range(0, N+1):
        initial_state = (j, N-j)
        final_dist = list(transition_matrix_power(edges, power=power, yield_all=False, initial_state=initial_state))[-1]
        
        for i in range(0, N+1):
            try:
                data.append((j,i, -math.log(final_dist[(i, N-i)])))
            except ValueError:
                data.append((j,i,0))
    
    heatmap.main(data, xfunc=int, yfunc=int, rounding=False)
    pyplot.xlim(0, N)
    pyplot.ylim(0, N)
    pyplot.xlabel("Initial State $(i, N-i)$")
    pyplot.ylabel("Final State $(j, N-j)$")
    pyplot.title("SI after %s iterations" % str(power))
    

def self_info_heatmap_movie(N=30, n=2, m=[[1,2],[2,1]], beta=1, max_power=3000, directory="self_info_heatmap_frames"): 

    for j in range(0, max_power):
        pyplot.clf()
        self_info_heatmap(N=N, n=n, m=m, beta=beta, power=j)
        pyplot.title("Iteration %s" % str(j))
        digits = len(str(max_power))
        filename = os.path.join(directory, ensure_digits(digits, str(j)) + ".png")
        print j, "saving", filename
        pyplot.savefig(filename, dpi=250)
    
    framerate = 20
    movie_filename = "movie.mp4"
    # avconv -r 20 -i "%04d.png" -b:v 1000k test.mp4
    subprocess.call(["avconv", "-r", '\"' + str(framerate), "-i", directory + "/" + "%0" + str(digits) + "d.png" + '\"', "-b:v", "1000k", movie_filename] )

def skewness_si_figure():
    N = 80
    beta = 1.
    m = [[1,2], [2,1]] # hawk-dove
    #m = [[1,3],[2,1]]
    #m = [[2,1],[1,2]] # coordination
    #m = [[2,2],[1,1]] # PD

    grid_spec = matplotlib.gridspec.GridSpec(1, 2)
    grid_spec.update(hspace=0.5)
    
    ax1 = pyplot.subplot(grid_spec[0, 0])
    
    #pyplot.figure()
    skewness_heatmap(N=N, m=m, beta=beta)
    
    ax1 = pyplot.subplot(grid_spec[0, 1])
    
    #pyplot.figure()
    self_info_heatmap(N=N, m=m, power=4*N, beta=beta)
    pyplot.show()

if __name__ == '__main__':
    #N = 60
    #beta = 1.
    #m = [[1,2], [2,1]] # hawk-dove
    ##m = [[1,3],[2,1]]
    ##m = [[2,1],[1,2]] # coordination
    ##m = [[2,2],[1,1]] # PD

    #pyplot.figure()
    #skewness_heatmap(N=N, m=m, beta=beta)
    #pyplot.figure()
    #self_info_heatmap(N=N, m=m, power=4*N, beta=beta)
    #pyplot.show()
    
    #self_info_plot(N=30, m=m, power=10)
    #self_info_movie(N=30, m=m, max_power=2000, beta=2.)    
    #N=30
    #self_info_heatmap(N=N, m=m, power=N)
    #self_info_heatmap_movie(N=N, m=m, max_power=10*N)
    
    skewness_si_figure()
    exit()


