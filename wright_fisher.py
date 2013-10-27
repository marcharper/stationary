
from math_helpers import kl_divergence, simplex_generator, one_step_indicies_generator, dot_product, normalize, q_divergence


import numpy
from numpy import array, log, exp

from scipy.special import gammaln
from scipy.misc import logsumexp

#####################
### Wright-Fisher ###
#####################

#Memoize multinomial coefficients

#M = numpy.zeros((N,N))
#for i in range(N+1):
    #for j in range(N+1-i):
        #M[i][j] = gammaln(N+1) - gammaln(i+1) - gammaln(j+1) - gammaln(k+1)

##def log_factorial(x):
    ##"""Returns the logarithm of x!
    ##Also accepts lists and NumPy arrays in place of x."""
    ##return gammaln(array(x)+1)

#def multinomial_probability(xs, ps):
    #n = sum(xs)
    #xs, ps = array(xs), array(ps)
    ##result = log_factorial(n) - sum(log_factorial(xs)) + sum(xs * log(ps))
    #result = M[xs[0]][xs[1]] + sum(xs * log(ps))
    #return exp(result)

### Works but has enormous memory footprint, ~O(N^4)
#def multivariate_wf_transitions(N, incentive, mu=0.001):
    #"""Computes transitions for dimension n=3 moran process given a game matrix."""
    #edges = []
    #for current_state in simplex_generator(N):
        #for next_state in simplex_generator(N):
            #inc = incentive(current_state)
            #ps = []
            #s = float(sum(inc))
            #r = dot_product(inc, [1-2*mu, mu, mu]) / s
            #ps.append(r)
            #r = dot_product(inc, [mu, 1-2*mu, mu]) / s
            #ps.append(r)
            #r = dot_product(inc, [mu, mu, 1-2*mu]) / s
            #ps.append(r)
            #edges.append((current_state, next_state, multinomial_probability(next_state, ps)))            
    #return edges

#def multivariate_wf_transitions(N, incentive, mu=0.001):
    #M = numpy.zeros(shape=(N+1,N+1))
    #for i,j,k in simplex_generator(N):
        #M[i][j] = gammaln(N+1) - gammaln(i+1) - gammaln(j+1) - gammaln(k+1)

    ##def log_factorial(x):
        ##"""Returns the logarithm of x!
        ##Also accepts lists and NumPy arrays in place of x."""
        ##return gammaln(array(x)+1)

    #def multinomial_probability(xs, ps):
        #n = sum(xs)
        #xs, ps = array(xs), array(ps)
        ##result = log_factorial(n) - sum(log_factorial(xs)) + sum(xs * log(ps))
        #result = M[xs[0]][xs[1]] + sum(xs * log(ps))
        #return exp(result)

    #"""Computes transitions for dimension n=3 moran process given a game matrix."""
    #def g(current_state, next_state):
        #inc = incentive(current_state)
        #ps = []
        #s = float(sum(inc))
        #r = dot_product(inc, [1-2*mu, mu, mu]) / s
        #ps.append(r)
        #r = dot_product(inc, [mu, 1-2*mu, mu]) / s
        #ps.append(r)
        #r = dot_product(inc, [mu, mu, 1-2*mu]) / s
        #ps.append(r)
        #return multinomial_probability(next_state, ps)
    #return g

def cache_M(N, num_types=3):
    M = numpy.zeros(shape=(N+1,N+1))
    for i,j,k in simplex_generator(N, d=num_types-1):
        M[i][j] = gammaln(N+1) - gammaln(i+1) - gammaln(j+1) - gammaln(k+1)
    return M

### Works but has somewhat less enormous memory footprint, ~O(N^4)
def multivariate_transitions(N, incentive, mu=0.001, num_types=3):
    """Computes transitions for dimension n=3 moran process given a game matrix."""
    M = cache_M(N)

    def multinomial_probability(xs, ps):
        n = sum(xs)
        xs, ps = array(xs), array(ps)
        #result = log_factorial(n) - sum(log_factorial(xs)) + sum(xs * log(ps))
        result = M[xs[0]][xs[1]] + sum(xs * log(ps))
        return exp(result)
    
    edges = numpy.zeros(shape=(N+1,N+1,N+1,N+1))
    for current_state in simplex_generator(N, num_types-1):
        for next_state in simplex_generator(N, num_types-1):
            inc = incentive(current_state)
            ps = []
            s = float(sum(inc))
            if s == 0:
                print "##### s is zero!!!"
            r = dot_product(inc, [1-2*mu, mu, mu]) / s
            ps.append(r)
            r = dot_product(inc, [mu, 1-2*mu, mu]) / s
            ps.append(r)
            r = dot_product(inc, [mu, mu, 1-2*mu]) / s
            ps.append(r)
            edges[current_state[0]][current_state[1]][next_state[0]][next_state[1]] = multinomial_probability(next_state, ps)
            #edges.append((current_state, next_state, multinomial_probability(next_state, ps)))            
            
    def g(current_state, next_state):
        return edges[current_state[0]][current_state[1]][next_state[0]][next_state[1]]
    return g
            
#### Log-space
#def log_multivariate_wf_transitions(N, incentive, mu=0.001):
    #"""Computes transitions for dimension n=3 moran process given a game matrix."""
    #M = cache_M(N)

    #def multinomial_probability(xs, ps):
        #n = sum(xs)
        #xs, ps = array(xs), array(ps)
        ##result = log_factorial(n) - sum(log_factorial(xs)) + sum(xs * log(ps))
        ##result = M[xs[0]][xs[1]] + sum(xs * log(ps))
        #result = logsumexp([M[xs[0]][xs[1]], logsumexp(ps, b=xs)])
        #return exp(result)
    
    #edges = numpy.zeros(shape=(N+1,N+1,N+1,N+1))
    #for current_state in simplex_generator(N):
        #for next_state in simplex_generator(N):
            #inc = incentive(current_state)
            #ps = []
            #s = float(logsumexp(inc))
            #r = exp( logsumexp(inc, b=[1-2*mu, mu, mu]) - s)
            #ps.append(r)
            #r = exp( logsumexp(inc, b=[mu, 1-2*mu, mu]) - s)
            #ps.append(r)
            #r = exp( logsumexp(inc, b=[mu, mu, 1-2*mu]) - s)
            #ps.append(r)
            #edges[current_state[0]][current_state[1]][next_state[0]][next_state[1]] = multinomial_probability(next_state, ps)
            ##edges.append((current_state, next_state, multinomial_probability(next_state, ps)))            
            
    #def g(current_state, next_state):
        #return edges[current_state[0]][current_state[1]][next_state[0]][next_state[1]]

    #return g

def stationary_distribution(N, edge_func, iterations=100, convergence_lim=1e-13):
    """n=3 sparse matrix approach."""
    states = list(simplex_generator(N))
    ranks = dict(zip(states, [1./float(len(states))]*(len(states))))
    for iteration in range(iterations):
        if iteration > 100:
            if iteration % 50:
                s = kl_divergence(ranks, previous_ranks)
                if s < convergence_lim:
                    break
        new_ranks = dict()
        for x in simplex_generator(N):
            new_rank = 0
            for y in simplex_generator(N):
                w = edge_func(y,x)
                new_rank += w * ranks[y]
            new_ranks[x] = new_rank
        previous_ranks = ranks
        ranks = new_ranks
    d = dict()
    for m, r in ranks.items():
        i,j,k = m
        d[(i,j,k)] = r
    return d

def kl(N, edge_func, q_d=1):
    e = dict()
    dist = q_divergence(q_d)
    for x in simplex_generator(N):
        e[x] = 0.
        for y in simplex_generator(N):
            w = edge_func(x,y)
            # could use the fact that E(x) = n p here instead for efficiency, but this is relatively fast compared to the stationary calculation already
            e[x] += numpy.array(y) * w
    d = dict()
    for (i, j, k), v in e.items():
        # KL doesn't play well with boundary states.
        if i*j*k == 0:
            continue
        d[(i,j,k)] = dist(normalize(v), normalize([i,j,k]))
    return d
