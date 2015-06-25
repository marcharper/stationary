
import incentives
import incentive_process
import wright_fisher

def bomze_matrices(filename="bomze.txt"):
    """
    Yields the 48 matrices from I.M. Bomze's classification of three player phase
    portraits.
    """

    handle = open(filename)
    for line in handle:
        a,b,c,d,e,f,g,h,i = map(float, line.split())
        yield [[a,b,c],[d,e,f],[g,h,i]]