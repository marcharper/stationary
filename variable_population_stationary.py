import math

from matplotlib import pyplot

import ternary

from stationary.processes.variable_population_size import variable_population_transitions, even_death, kl
from stationary.processes.incentive_process import linear_fitness_landscape, replicator

from stationary import stationary_distribution


if __name__ == '__main__':
    N = 40
    mu = 3./2. * 1./N
    m = [[1, 2], [2, 1]]
    fitness_landscape = linear_fitness_landscape(m, normalize=False)
    incentive = replicator(fitness_landscape)
    death_probabilities = even_death(N)

    edges = variable_population_transitions(N, fitness_landscape, death_probabilities, incentive=incentive, mu=mu)
    s = stationary_distribution(edges, iterations=10000)

    # Print out the states with the highest stationary probabilities
    vs = [(v,k) for (k,v) in s.items()]
    vs.sort(reverse=True)
    print vs[:10]

    # Plot the stationary distributoin and expected divergence

    figure, tax = ternary.figure(scale=N)
    tax.heatmap(s)

    d = kl(N, edges, q=0, func=math.sqrt)

    figure, tax = ternary.figure(scale=N)
    tax.heatmap(d)

    pyplot.show()
