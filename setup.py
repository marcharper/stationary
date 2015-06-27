
import os
from setuptools import setup

# Utility function to read the README file.
# Used for the long_description.  It's nice, because now 1) we have a top level
# README file and 2) it's easier to type in the README file than to put a raw
# string in below ...

def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

setup(
    name = "stationary",
    version = "0.0.1",
    author = "Marc Harper",
    author_email = "marc.harper@gmail.com",
    description = "Stationary distributions for finite Markov processes",
    license = "MIT",
    keywords = "markov stationary",
    url = "https://github.com/marcharper/stationary",
    packages=['stationary'],
    install_requires=['numpy', 'scipy', 'matplotlib', 'nose'],
    long_description=read('README.md'),
)