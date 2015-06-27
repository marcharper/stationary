
import os

def bomze_matrices(filename="bomze.txt"):
    """
    Yields the 48 matrices from I.M. Bomze's classification of three player phase
    portraits.
    """

    this_dir, this_filename = os.path.split(__file__)

    handle = open(os.path.join(this_dir, filename))
    for line in handle:
        a,b,c,d,e,f,g,h,i = map(float, line.split())
        yield [[a,b,c],[d,e,f],[g,h,i]]
