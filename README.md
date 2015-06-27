
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
offloaded to a C++ implementation (faster and smaller memory footpring).

Examples
--------

There are many examples in the test suite and some more complex examples in the
following files:

- [test_stationary.py](https://github.com/marcharper/stationary/blob/master/tests/test_stationary.py): has a huge number of examples
- [examples.py[(https://github.com/marcharper/stationary/blob/master/examples.py): a number of simple examples of various types
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