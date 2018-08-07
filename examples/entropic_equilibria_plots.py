"""Figures for the publication
"Entropic Equilibria Selection of Stationary Extrema in Finite Populations"
"""

from __future__ import print_function
import math
import os
import pickle
import sys

import matplotlib
from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import scipy.misc
import ternary

import stationary
from stationary.processes import incentives, incentive_process


## Global Font config for plots ###
font = {'size': 14}
matplotlib.rc('font', **font)


def compute_entropy_rate(N=30, n=2, m=None, incentive_func=None, beta=1.,
                         mu=None, exact=False, lim=1e-13, logspace=False):
    if not m:
        m = np.ones((n, n))
    if not incentive_func:
        incentive_func = incentives.fermi
    if not mu:
        # mu = (n-1.)/n * 1./(N+1)
        mu = 1. / N

    fitness_landscape = incentives.linear_fitness_landscape(m)
    incentive = incentive_func(fitness_landscape, beta=beta, q=1)
    edges = incentive_process.multivariate_transitions(
        N, incentive, num_types=n, mu=mu)
    s = stationary.stationary_distribution(edges, exact=exact, lim=lim,
                                           logspace=logspace)
    e = stationary.entropy_rate(edges, s)
    return e, s


# Entropy Characterization Plots

def dict_max(d):
    k0, v0 = list(d.items())[0]
    for k, v in d.items():
        if v > v0:
            k0, v0 = k, v
    return k0, v0


def plot_data_sub(domain, plot_data, gs, labels=None, sci=True, use_log=False):
    # Plot Entropy Rate
    ax1 = plt.subplot(gs[0, 0])
    ax1.plot(domain, [x[0] for x in plot_data[0]], linewidth=2)

    # Plot Stationary Probabilities and entropies
    ax2 = plt.subplot(gs[1, 0])
    ax3 = plt.subplot(gs[2, 0])

    if use_log:
        transform = math.log
    else:
        transform = lambda x: x

    for i, ax, t in [(1, ax2, lambda x: x), (2, ax3, transform)]:
        if labels:
            for data, label in zip(plot_data, labels):
                ys = list(map(t, [x[i] for x in data]))
                ax.plot(domain, ys, linewidth=2, label=label)
        else:
            for data in plot_data:
                ys = list(map(t, [x[i] for x in data]))
                ax.plot(domain, ys, linewidth=2)

    ax1.set_ylabel("Entropy Rate")
    ax2.set_ylabel("Stationary\nExtrema")
    if use_log:
        ax3.set_ylabel("log RTE $H_v$")
    else:
        ax3.set_ylabel("RTE $H_v$")
    if sci:
        ax2.yaxis.get_major_formatter().set_powerlimits((0, 0))
        ax3.yaxis.get_major_formatter().set_powerlimits((0, 0))
    return ax1, ax2, ax3


def ER_figure_beta2(N, m, betas):
    """Varying Beta, two dimensional example"""
    # Beta test
    # m = [[1, 4], [4, 1]]

    # Compute the data
    ss = []
    plot_data = [[]]

    for beta in betas:
        print(beta)
        e, s = compute_entropy_rate(N=N, m=m, beta=beta, exact=True)
        ss.append(s)
        state, s_max = dict_max(s)
        plot_data[0].append((e, s_max, e / s_max))

    gs = gridspec.GridSpec(3, 2)
    ax1, ax2, ax3 = plot_data_sub(betas, plot_data, gs, sci=False)
    ax3.set_xlabel("Strength of Selection $\\beta$")

    # Plot stationary distribution
    ax4 = plt.subplot(gs[:, 1])
    for s in ss[::4]:
        ax4.plot(range(0, N+1), [s[(i, N-i)] for i in range(0, N+1)])
    ax4.set_title("Stationary Distributions")
    ax4.set_xlabel("Population States $(i , N - i)$")


def remove_boundary(s):
    s1 = dict()
    for k, v in s.items():
        a, b, c = k
        if a * b * c != 0:
            s1[k] = v
    return s1


def ER_figure_beta3(N, m, mu, betas, iss_states, labels, stationary_beta=0.35,
                    pickle_filename="figure_beta3.pickle"):
    """Varying Beta, three dimensional example"""

    ss = []
    plot_data = [[] for _ in range(len(iss_states))]

    if os.path.exists(pickle_filename):
        with open(pickle_filename, 'rb') as f:
            plot_data = pickle.load(f)
    else:
        for beta in betas:
            print(beta)
            e, s = compute_entropy_rate(
                N=N, m=m, n=3, beta=beta, exact=False, mu=mu, lim=1e-10)
            ss.append(s)
            for i, iss_state in enumerate(iss_states):
                s_max = s[iss_state]
                plot_data[i].append((e, s_max, e / s_max))
        with open(pickle_filename, 'wb') as f:
            pickle.dump(plot_data, f)

    gs = gridspec.GridSpec(3, 2)

    ax1, ax2, ax3 = plot_data_sub(betas, plot_data, gs, labels=labels,
                                  use_log=True, sci=False)
    ax3.set_xlabel("Strength of selection $\\beta$")
    ax2.legend(loc="upper right")

    # Plot example stationary
    ax4 = plt.subplot(gs[:, 1])
    _, s = compute_entropy_rate(
        N=N, m=m, n=3, beta=stationary_beta, exact=False, mu=mu, lim=1e-15)
    _, tax = ternary.figure(ax=ax4, scale=N,)
    tax.heatmap(s, cmap="jet", style="triangular")
    tax.ticks(axis='lbr', linewidth=1, multiple=10, offset=0.015)
    tax.clear_matplotlib_ticks()
    ax4.set_xlabel("Population States $a_1 + a_2 + a_3 = N$")

    # tax.left_axis_label("$a_1$")
    # tax.right_axis_label("$a_2$")
    # tax.bottom_axis_label("$a_3$")


def ER_figure_N(Ns, m, beta=1, labels=None):
    """Varying population size."""

    ss = []
    plot_data = [[] for _ in range(3)]
    n = len(m[0])

    for N in Ns:
        print(N)
        mu = 1 / N
        norm = float(scipy.misc.comb(N+n, n))
        e, s = compute_entropy_rate(
            N=N, m=m, n=3, beta=beta, exact=False, mu=mu, lim=1e-10)
        ss.append(s)
        iss_states = [(N, 0, 0), (N / 2, N / 2, 0), (N / 3, N / 3, N / 3)]
        for i, iss_state in enumerate(iss_states):
            s_max = s[iss_state]
            plot_data[i].append((e, s_max, e / (s_max * norm)))
    # Plot data
    gs = gridspec.GridSpec(3, 1)
    ax1, ax2, ax3 = plot_data_sub(Ns, plot_data, gs, labels, use_log=True, sci=False)
    ax2.legend(loc="upper right")
    ax3.set_xlabel("Population Size $N$")


def ER_figure_mu(N, mus, m, iss_states, labels, beta=1.,
                 pickle_filename="figure_mu.pickle"):
    """
    Plot entropy rates and trajectory entropies for varying mu.
    """
    # Compute the data
    ss = []
    plot_data = [[] for _ in range(len(iss_states))]

    if os.path.exists(pickle_filename):
        with open(pickle_filename, 'rb') as f:
            plot_data = pickle.load(f)
    else:
        for mu in mus:
            print(mu)
            e, s = compute_entropy_rate(
                N=N, m=m, n=3, beta=beta, exact=False, mu=mu, lim=1e-10,
                logspace=True)
            ss.append(s)
            for i, iss_state in enumerate(iss_states):
                s_max = s[iss_state]
                plot_data[i].append((e, s_max, e / s_max))
        with open(pickle_filename, 'wb') as f:
            pickle.dump(plot_data, f)

    # Plot data
    gs = gridspec.GridSpec(3, 1)
    gs.update(hspace=0.5)
    ax1, ax2, ax3 = plot_data_sub(mus, plot_data, gs, labels, use_log=True)
    ax2.legend(loc="upper right")
    ax3.set_xlabel("Mutation rate $\mu$")


if __name__ == '__main__':
    fig_num = sys.argv[1]

    if fig_num == "1":
        ## Figure 1
        # Varying beta, two dimensional
        N = 30
        m = [[1, 2], [2, 1]]
        betas = np.arange(0, 8, 0.2)
        ER_figure_beta2(N, m, betas)
        plt.tight_layout()
        plt.show()

    if fig_num == "2":
        ## Figure 2
        # # Varying beta, three dimensional
        N = 60
        mu = 1. / N
        m = [[0, 1, 1], [1, 0, 1], [1, 1, 0]]
        iss_states = [(N, 0, 0), (N / 2, N / 2, 0), (N / 3, N / 3, N / 3)]
        labels = ["$v_0$", "$v_1$", "$v_2$"]
        betas = np.arange(0.02, 0.6, 0.02)
        ER_figure_beta3(N, m, mu, betas, iss_states, labels)
        plt.show()

    if fig_num == "3":
        ## Figure 3
        # Varying mutation rate figure
        N = 42
        mus = np.arange(0.0001, 0.015, 0.0005)
        m = [[0, 1, 1], [1, 0, 1], [1, 1, 0]]
        iss_states = [(N, 0, 0), (N / 2, N / 2, 0), (N / 3, N / 3, N / 3)]
        labels = ["$v_0$: (42, 0, 0)", "$v_1$: (21, 21, 0)", "$v_2$: (14, 14, 14)"]
        # labels = ["$v_0$", "$v_1$", "$v_2$"]
        ER_figure_mu(N, mus, m, iss_states, labels, beta=1.)
        plt.show()

    if fig_num == "4":
        ## Figure 4
        # Note: The RPS landscape takes MUCH longer to converge!
        # Consider using the C++ implementation instead for larger N.
        N = 120  # Manuscript uses 180
        mu = 1. / N
        m = incentives.rock_paper_scissors(a=-1, b=-1)
        _, s = compute_entropy_rate(
            N=N, m=m, n=3, beta=1.5, exact=False, mu=mu, lim=1e-16)
        _, tax = ternary.figure(scale=N)
        tax.heatmap(remove_boundary(s), cmap="jet", style="triangular")
        tax.ticks(axis='lbr', linewidth=1, multiple=60)
        tax.clear_matplotlib_ticks()
        plt.show()

    if fig_num == "5":
        # ## Figure 5
        # Varying Population Size
        Ns = range(6, 6*6, 6)
        m = [[0, 1, 1], [1, 0, 1], [1, 1, 0]]
        labels = ["$v_0$", "$v_1$", "$v_2$"]
        ER_figure_N(Ns, m, beta=1, labels=labels)
        plt.show()

