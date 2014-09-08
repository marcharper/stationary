import subprocess

from incentives import linear_fitness_landscape, fermi
import incentive_process

import matplotlib
from matplotlib import pyplot
from three_dim import heatmap


##################
# Export to C++ ##
##################

#def output_enumerated_edges(edges, iterations=None, convergence_lim=1e-8):
def output_enumerated_edges(edges, filename="enumerated_edges.csv"):
    """Essentially raising the transition probabilities matrix to a large power using a sparse-matrix implementation."""
    all_states = set()
    for (source, target, weight) in edges:
        all_states.add(source)
        all_states.add(target)
    sorted_edges = list(sorted(all_states))

    enum = dict()
    inv_enum = []
    for index, state in enumerate(all_states):
        enum[state] = index
        inv_enum.append(state)
    
    ## output enumerated_edges
    with open(filename, 'w') as outfile:
        #outfile.write(str(sum(edges[0][0])) + "\n")
        outfile.write(str(len(all_states)) + "\n")
        outfile.write(str(len(edges[0][0])) + "\n")
        for (source, target, weight) in edges:
            row = [str(enum[source]), str(enum[target]), str.format('%.40f' % weight)]
            #row = map(str,[enum[source], enum[target], weight])
            outfile.write(",".join(row) + "\n")
    return inv_enum

def rsp_N_test(N=60, q=1, beta=1., convergence_lim=1e-9, mu=None):
    #m = [[0, -1, 1], [1, 0, -1], [-1, 1, 0]]
    m = [[0,1,1],[1,0,1],[1,1,0]]
    num_types = len(m[0])
    fitness_landscape = linear_fitness_landscape(m)
    if not mu:
        mu = 3./2*1./N
    incentive = fermi(fitness_landscape, beta=beta, q=q)
    edges = incentive_process.multivariate_transitions(N, incentive, num_types=num_types, mu=mu)

    #d = approximate_stationary_distribution(edges, convergence_lim=convergence_lim)

    inv_enum = output_enumerated_edges(edges)

    # Call out to C++
    subprocess.call(["./a.out"])

    ## Load Stationary and reverse enumeration
    ranks = load_stationary()
    d = dict()
    for s, v in ranks:
        print s, v
        state = inv_enum[s]
        d[state] = v

    #for k, v in d.items():
        #print k,v

    # plot
    grid_spec = matplotlib.gridspec.GridSpec(1, 1)
    grid_spec.update(hspace=0.5)
    ax1 = pyplot.subplot(grid_spec[0, 0])
    heatmap(d, ax=ax1, scientific=True)
    pyplot.show()
    

def load_stationary(filename="enumerated_stationary.txt"):
    s = []
    with open(filename) as input_file:
        for line in input_file:
            line = line.strip()
            state, value = line.split(',')
            #print state, value
            s.append((int(state), float(value)))
    return s


if __name__ == '__main__':
    N = 40
    mu = 1./N * 3./2
    rsp_N_test(N=N, beta=1., mu=mu, convergence_lim=1e-14)
