import numpy
from numpy import log, exp, arange

try:
    from scipy.misc import logsumexp
except ImportError:
    from numpy import logaddexp
    logsumexp = logaddexp.reduce

from scipy.special import gammaln

def slice_dictionary(d, N, slice_index=0, slice_value=0):
    """
    Take a three dimensional slice from a four dimensional
    dictionary.
    """

    slice_dict = dict()
    for state in simplex_generator(N, 2):
        new_state = list(state)
        new_state.insert(slice_index, slice_value)
        slice_dict[state] = d[tuple(new_state)]
    return slice_dict

def squared_error(d1, d2):
    """
    Compute the squared error between two vectors.
    """

    s = 0.
    for k in range(len(d1)):
        s += (d1[k] - d2[k])**2
    return numpy.sqrt(s)

def squared_error_dict(d1, d2):
    """
    Compute the squared error between two vectors, stored as dictionaries.
    """

    s = 0.
    for k in d1.keys():
        s += (d1[k] - d2[k])**2
    return numpy.sqrt(s)

## Vectors

def multiply_vectors(a, b):
    c = []
    for i in range(len(a)):
        c.append(a[i]*b[i])
    return c

def dot_product(a, b):
    c = 0
    for i in range(len(a)):
        c += a[i] * b[i]
    return c

# vector
def normalize(x):
    s = float(sum(x))
    for j in range(len(x)):
        x[j] /= s
    return x

# dictionary
def normalize_dictionary(x):
    s = float(sum(x.values()))
    for k in x.keys():
        x[k] /= s
    return x

# Various factorials

def inc_factorial(x, n):
    p = 1.
    for i in range(0, n):
       p *= (x + i)
    return p

def factorial(i):
    p = 1.
    for j in range(2, i+1):
        p *= j
    return p

def log_inc_factorial(x,n):
    p = 1.
    for i in range(0, n):
       p += log(x + i)
    return p

# Could also use gammaln
def log_factorial(i):
    p = 1.
    for j in range(2, i+1):
        p += log(j)
    return p

## Simplex discretizers

def simplex_generator(N, d=2):
    """
    Generates a discretation of the simplex.

    Parameters
    ----------
    N: int
        The number of subdivsions in each dimension
    d: int, 2
        The dimension of the simplex (the number of population types is d+1

    Yields
    ------
    (d+1)-tuples of numbers summing to N. The total number of yielded tuples is
    equal to the simplicial polytopic number corresponding to N and d,
    binom{N + d - 1}{d} (see https://en.wikipedia.org/wiki/Figurate_number )
    """

    if d == 1:
        for i in range(N+1):
            yield (i,N-i)
    if d > 1:
        for j in range(N+1):
            for s in simplex_generator(N-j, d-1):
                t = [j]
                t.extend(s)
                yield tuple(t)

def one_step_generator(d):
    """
    Generates the arrays needed to construct neighboring states one step away
    from a state in the dimension d simplex.
    """

    if d == 1:
        yield [1, -1]
        yield [-1, 1]
        return
    for plus_index in range(d+1):
        for minus_index in range(d+1):
            if minus_index == plus_index:
                continue
            step = [0]*(d+1)
            step[plus_index] = 1
            step[minus_index] = -1
            yield step

def one_step_indicies_generator(d):
    """
    Generates the indices that form all the neighboring states, by adding +1 in
    one index and -1 in another.
    """
    if d == 1:
        yield [0, 1]
        yield [1, 0]
        return
    for plus_index in range(d+1):
        for minus_index in range(d+1):
            if minus_index == plus_index:
                continue
            yield (plus_index, minus_index)

#def enumerate_states(N, d):
    #"""d is the dimension, the number of types is d+1."""
    #enum = dict()
    #inv = []
    #for i, state in enumerate(simplex_generator(N, d)):
        #enum[state] = i
        #inv.append(state)

## Information Theory

def kl_divergence(p, q):
    """
    Computes the KL-divergence or relative entropy of to input distributions.

    Parameters
    ----------
    p, q: lists
        The probability distributions to compute the KL-divergence for

    Returns
    -------
    float, the KL-divergence of p and q
    """

    s = 0.
    for i in range(len(p)):
        if p[i] == 0:
            continue
        if q[i] == 0:
            return float('nan')
        try:
            s += p[i] * log(p[i])
        except (ValueError, ZeroDivisionError):
            continue
        try:
            s -= p[i] * log(q[i])
        except (ValueError, ZeroDivisionError):
            continue
    return s

def kl_divergence_dict(p, q):
    """
    Computes the KL-divergence of distributions given as dictionaries.
    """
    s = 0.

    p_list = []
    q_list = []

    for i in p.keys():
        p_list.append(p[i])
        q_list.append(q[i])
    return kl_divergence(p_list, q_list)

def q_divergence(q):
    """
    Returns the divergence function corresponding to the parameter value q. For
    q == 0 this function is one-half the squared Euclidean distance. For q == 1
    this function returns the KL-divergence.
    """

    if q == 0:
        def d(x, y):
            return 0.5 * numpy.dot((x-y),(x-y))
        return d
    if q == 1:
        return kl_divergence
    if q == 2:
        def d(x,y):
            s = 0.
            for i in range(len(x)):
                s += log(x[i] / y[i]) + 1 - x[i] / y[i]
            return -s
        return d
    q = float(q)
    def d(x, y):
        s = 0.
        for i in range(len(x)):
            s += (numpy.power(y[i], 2 - q) - numpy.power(x[i], 2 - q)) / (2 - q)
            s -= numpy.power(y[i], 1 - q) * (y[i] - x[i])
        s = -s / (1 - q)
        return s
    return d

def shannon_entropy(p):
    s = 0.
    for i in range(len(p)):
        try:
            s += p[i] * log(p[i])
        except ValueError:
            continue
    return -1.*s    

def binary_entropy(p):
    return -p*log(p) - (1-p) * log(1-p)
