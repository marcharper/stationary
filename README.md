
# Stationary

This is a python library for computing stationary distributions of finite Markov
process, with a particular emphasis on finite population dynamics.

The library can approximate solutions for arbitrary finite Markov processes and
exact stationary distributions for reversible Markov processes on discretized
simplices. For approximate calculations one need only supply a list of weighted
edges:

```
[(source_state, target_state, transition_probability), ...]
```

The states of the process can be any hashable python object (integers, strings,
etc.).

Included are functions to generate transition probabilities for the Moran process
with mutation and various generalizations, including Fermi processes, dynamics on
graphs, dynamics in populations of varying sizes, and the Wright-Fisher process.

For example, the following image (a stationary distribution for a
rock-paper-scissors dynamic on a population of size 560) was created with this 
library and a [ternary plotting library](https://github.com/marcharper/python-ternary):

<img src ="https://github.com/marcharper/python-ternary/blob/master/readme_images/heatmap_rsp.png" width="300" height="300"/>

For very large state spaces, the stationary distribution calculation can be
offloaded to a C++ implementation (faster and smaller memory footprint).

Calculation of Stationary Distributions
---------------------------------------

The library computes stationary distributions in a variety of ways. Transition
probabilities are represented by sparse matrices or by functions on the product
of the state space. The latter is useful when the transition matrix is too large
to fit into memory, such as the Wright-Fisher process for a population of size
N with n-types, which requires 
<img src="http://www.sciweavers.org/tex2img.php?eq=%5Cmathcal%7BO%7D%5Cleft%28N%5E%7B2%28n-1%29%7D%5Cright%29&bc=White&fc=Black&im=jpg&fs=12&ff=arev&edit=0" align="center" border="0" alt="\mathcal{O}\left(N^{2(n-1)}\right)" width="97" height="26" />

floating point values to specify the transition matrix.

The stationary distribution calculation function `stationary.stationary_distribution` takes
either a list of weighted edges (as above) or a function specifying the transitions
and the collection of states of the process. You can specify the following:

- Transitions or a function that computes transitions
- Compute in log-space with `logspace=True`, useful (necessary) for processes with very small
probabilities
- Compute the stationary distribution exactly or approximately with `exact=True` (default is false. If `False`, the library computes large powers of the transition matrix times an
initial state. If exact, the library attemps to use the following formula:

<img src="http://www.sciweavers.org/tex2img.php?eq=s%28v_k%29%20%3D%20s%28v_0%29%20%5Cprod_%7Bj%3D1%7D%5E%7Bk-1%7D%7B%20%5Cfrac%7BT%28v_j%2C%20v_%7Bj%2B1%7D%29%7D%7BT%28v_%7Bj%2B1%7D%2C%20v_%7Bj%7D%29%7D%7D&bc=White&fc=Black&im=jpg&fs=12&ff=arev&edit=0" align="center" border="0" alt="s(v_k) = s(v_0) \prod_{j=1}^{k-1}{ \frac{T(v_j, v_{j+1})}{T(v_{j+1}, v_{j})}}" width="229" height="56" />

This formula only works for reversible processes on the simplex -- a particular encoding
of states and paths is assumed.

The library can also compute exact solutions for the neutral fitness landscape for the
Moran process.

Examples
--------

Let's walk through a detailed example. For the classical Moran process, we have
a population of two types, A and B. Type B has relative fitness `r` versus the
fitness of type A (which is 1). It is well-known that the fixation probability
of type A is

<img src="http://www.sciweavers.org/tex2img.php?eq=%5Crho_A%20%3D%20%5Cfrac%7B1%20-%20r%5E%7B-1%7D%7D%7B1%20-%20r%5E%7B-N%7D%7D&bc=White&fc=Black&im=jpg&fs=12&ff=arev&edit=0" align="center" border="0" alt="\rho_A = \frac{1 - r^{-1}}{1 - r^{-N}}" width="107" height="47" />

It is also well know that the stationary distribution of the Moran process with
mutation converges to the following distribution when the mutation rate goes to
zero:

<img src="http://www.sciweavers.org/tex2img.php?eq=s%20%3D%20%5Cleft%28%5Cfrac%7B%5Crho_A%7D%7B%5Crho_A%20%2B%20%5Crho_B%7D%2C%200%2C%20%5Cldots%2C%200%2C%20%5Cfrac%7B%5Crho_B%7D%7B%5Crho_A%20%2B%20%5Crho_B%7D%5Cright%29&bc=White&fc=Black&im=jpg&fs=12&ff=arev&edit=0" align="center" border="0" alt="s = \left(\frac{\rho_A}{\rho_A + \rho_B}, 0, \ldots, 0, \frac{\rho_B}{\rho_A + \rho_B}\right)" width="267" height="47" />

where the stationary distribution is over the population states
[(0, N), (1, N-1), ..., (N, 0)].

In [fixation_examples.py](https://github.com/marcharper/stationary/blob/master/fixation_examples.py)
we compare the ratio of fixation probabilities with the stationary distribution
for small values of mu, `mu=10^-24`, producing the following plot:

![fixation_example.png](https://github.com/marcharper/stationary/blob/master/fixation_example.png)

In the top plot there is no visual distinction between the two values. The lower
plot has the difference in the two calculations, showing that the error is very
small.

More Examples
-------------

There are many examples in the test suite and some more complex examples in the
following files:

- [test_stationary.py](https://github.com/marcharper/stationary/blob/master/tests/test_stationary.py): has a huge number of examples
- [examples.py](https://github.com/marcharper/stationary/blob/master/examples.py): a number of simple examples of various types
- [cycle_stationary.py](https://github.com/marcharper/stationary/blob/master/cycle_stationary.py): a population process on a graph
- [variable_population_stationary.py](https://github.com/marcharper/stationary/blob/master/variable_population_stationary.py): a Moran process on a population of varying size

Unit Tests
----------

The library contains a number of tests to ensure that the calculations are
accurate, including comparisons to closed forms when available.

To run the suite of unit tests, use the command

```
nosetests -s tests
```

Note that there are many tests and some take a considerable amount of time
(several minutes in some cases).