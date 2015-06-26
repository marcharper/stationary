
from matplotlib import pyplot

import ternary

from stationary.utils.math_helpers import simplex_generator

def plot_stationary(s, ax=None):
    num_types = len(s.keys()[0])
    N = sum(s.keys()[0])

    if not ax:
        fig, ax = pyplot.subplots()


    if num_types == 2:
        domain = list(range(0, N+1))
        values = []
        for i in domain:
            state = (i, N-i)
            values.append(s[state])
        if not ax:
            fig, ax = pyplot.subplots()
        pyplot.plot(domain, values)
    
    if num_types == 3:
        fig, tax = ternary.figure(scale=N, ax=ax)
        tax.heatmap(s)

