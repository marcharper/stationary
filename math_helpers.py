#from math import log
import math
import numpy
from numpy import log, exp

def kl_divergence(p, q):
    s = 0.
    for i in range(len(p)):
        try:
            t = p[i] * math.log(p[i] / q[i])
            s += t
        except ValueError:
            continue
    return s

def q_divergence(q):
    """Returns the divergence function corresponding to the parameter value q."""
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
            s += (math.pow(y[i], 2 - q) - math.pow(x[i], 2 - q)) / (2 - q)
            s -= math.pow(y[i], 1 - q) * (y[i] - x[i])
        s = -s / (1 - q)
        return s
    return d

def arange(a, b, steps=100):
    """Similar to numpy.arange"""
    delta = (b - a) / float(steps)
    xs = []
    for i in range(steps):
        x = a + delta * i
        xs.append(x)
    return xs

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

#vector
def normalize(x):
    s = float(sum(x))
    for j in range(len(x)):
        x[j] /= s
    return x

#dictionary
def normalize_dictionary(x):
    s = float(sum(x.values()))
    for k in x.keys():
        x[k] /= s
    return x

## Information Theoretic Functions    
    
def kl_divergence(p, q):
    #print p, q
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
    #print p, q
    s = 0.
    for i in p.keys():
        if p[i] == 0:
            continue
        s += p[i] * log(p[i])
        s -= p[i] * log(q[i])

        #if q[i] == 0:
            ##return float('nan')
        #try:
            #s += p[i] * log(p[i])
        #except (ValueError, ZeroDivisionError):
            #continue
        #try:
            #s -= p[i] * log(q[i])
        #except (ValueError, ZeroDivisionError):
            #continue
    return s

#def shannon_entropy(p):
    #s = 0.
    #for i in range(len(p)):
        #try:
            #s += p[i] * log(p[i])
        #except ValueError:
            #continue
    #return -1.*s

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

def simplex_generator(N, d=2):
    """d is the dimension, the number of types is d+1."""
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
    """d is the dimension, the number of types is d+1."""
    if d == 1:
        yield [1,-1]
        yield [-1,1]
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
    if d == 1:
        yield [0, 1]
        yield [1,0]
        return
    for plus_index in range(d+1):
        for minus_index in range(d+1):
            if minus_index == plus_index:
                continue
            yield (plus_index, minus_index)

def edges_to_matrix(edges):
    # Enumerate states so we can put them in a matrix.
    all_states = set()
    for (a,b,v) in edges:
        all_states.add(a)
        all_states.add(b)
    enumeration = dict(zip(all_states, range(len(all_states))))
    # Build a matrix for the transitions
    mat = numpy.zeros((len(all_states), len(all_states)))
    for (a, b, v) in edges:
        mat[enumeration[a]][enumeration[b]] = v
    return mat, all_states, enumeration

