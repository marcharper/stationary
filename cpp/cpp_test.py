import cPickle as pickle
import subprocess

from incentives import linear_fitness_landscape, fermi
import incentive_process
from stationary import output_enumerated_edges

import matplotlib
from matplotlib import pyplot
from three_dim import heatmap

import scipy.misc

"""Todo:
enumerate simplex
turn compute edges into a generator
"""


##################
# Export to C++ ##
##################

def num_states(N, n=3):
    return scipy.misc.comb(N+n-1, n-1, exact=True)

#def enum_state(state):
    #i,j,k = state
    #N = i + j + k
    #e = (N+1) * i + j
    #return e

#def inv_enum_state(N, e):
    #k = N - e
    #j = e % (N+1)
    #i = (e - j) / (N+1)
    #return (i,j,k)

def rsp_N_test(N=60, q=1, beta=1., mu=None, pickle_filename="inv_enum.pickle", filename="enumerated_edges.csv"):
    m = [[0, -1, 1], [1, 0, -1], [-1, 1, 0]]
    #m = [[0,1,1],[1,0,1],[1,1,0]]
    num_types = len(m[0])
    #fitness_landscape = linear_fitness_landscape(m)
    if not mu:
        mu = 3./2*1./N
    edges = incentive_process.compute_edges(N, num_types, m=m, incentive_func=fermi, mu=mu, beta=beta)
    #d = approximate_stationary_distribution(edges, convergence_lim=convergence_lim)

    inv_enum = output_enumerated_edges(N, num_types, edges, filename=filename)
    # Pickle it
    with open(pickle_filename, 'wb') as output_file:
        pickle.dump(inv_enum, output_file)

    #return inv_enum
    # Call out to C++
    #subprocess.call(["./a.out", filename, "25000"])

def render_stationary():
    d = dict()
    for state, value in stationary_gen():
        d[state] = value
    # plot
    grid_spec = matplotlib.gridspec.GridSpec(1, 1)
    grid_spec.update(hspace=0.5)
    ax1 = pyplot.subplot(grid_spec[0, 0])
    heatmap(d, ax=ax1, scientific=True)
    pyplot.show()

def load_pickled_inv_enum(filename="inv_enum.pickle"):
    with open(filename, 'rb') as input_file:
        inv_enum = pickle.load(input_file)
    return inv_enum

def load_stationary_gen(filename="enumerated_stationary.txt"):
    s = []
    with open(filename) as input_file:
        for line in input_file:
            line = line.strip()
            state, value = line.split(',')
            yield (int(state), float(value))

def stationary_gen(filename="enumerated_stationary.txt", pickle_filename="inv_enum.pickle"):
    ## Load Stationary and reverse enumeration
    inv_enum = load_pickled_inv_enum(filename=pickle_filename)
    gen = load_stationary_gen(filename=filename)
    for enum_state, value in gen:
        state = inv_enum[enum_state]
        yield (state, value)

if __name__ == '__main__':
    #print num_states(600, 3)
    #exit()
    N = 1200 # takes 1.2 GB of memory to generate edges, too much to render plot
    mu = 1./N * 3./2
    beta = 1.
    rsp_N_test(N=N, beta=beta, mu=mu)
    #render_stationary()
    
