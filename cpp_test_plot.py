import cPickle as pickle
import math
import subprocess

import matplotlib
from matplotlib import pyplot

from ternary import colormapper, triangle_coordinates
from incentives import linear_fitness_landscape, fermi
import incentive_process
from three_dim import heatmap

"""Todo:
enumerate simplex
"""

def load_pickled_inv_enum(filename="inv_enum.pickle"):
    with open(filename, 'rb') as input_file:
        inv_enum = pickle.load(input_file)
    return inv_enum

def load_stationary_gen(filename="enumerated_stationary.txt"):
    s = []
    with open(filename) as input_file:
        for line in input_file:
            line = line.strip()
            state, value = line.split(',')
            yield (int(state), float(value))

def stationary_max_min(filename="enumerated_stationary.txt", pickle_filename="inv_enum.pickle", boundary=False):
    min_ = 1.
    max_ = 0.
    inv_enum = load_pickled_inv_enum(filename=pickle_filename)
    gen = load_stationary_gen(filename=filename)
    for enum_state, value in gen:
        state = inv_enum[enum_state]
        i,j,k = state
        if not boundary:
            if i*j*k == 0:
                continue
        if value > max_:
            max_ = value
        if value < min_:
            min_ = value
    return min_, max_

def stationary_gen(filename="enumerated_stationary.txt", pickle_filename="inv_enum.pickle"):
    ## Load Stationary and reverse enumeration
    inv_enum = load_pickled_inv_enum(filename=pickle_filename)
    gen = load_stationary_gen(filename=filename)
    for enum_state, value in gen:
        state = inv_enum[enum_state]
        yield (state, value)

def svg_triangle(coordinates, color):
    coord_str = []
    for c in coordinates:
        coord_str.append(",".join(map(str, c)))
    coord_str = " ".join(coord_str)
    rgb_str = color
    polygon = '<polygon points="%s" style="fill:%s;stroke%s;stroke-width:0"/>\n' % (coord_str, rgb_str, rgb_str)
    return polygon

def stationary_svg(N=120, boundary=False, svg_filename="stationary_heatmap.svg"):
    d = dict()
    output_file = open(svg_filename, 'w')
    height = N*math.sqrt(3)/2 + 2
    output_file.write('<svg height="%s" width="%s">\n' % (height, N))
    min_, max_ = stationary_max_min()
    print min_, max_
    for state, value in stationary_gen():
        i,j,k = state
        if not boundary:
            if i*j*k == 0:
                continue
        steps = i+j+k
        d[(i,j)] = value
        coordinates = triangle_coordinates(i,j)
        hex_color = colormapper(value, min_, max_)
        output_file.write(svg_triangle(coordinates, hex_color))
    # Draw smoothing triangles
    offset = 0
    if not boundary:
        offset = 1
    for i in range(offset, steps+1-offset):
        for j in range(offset, steps -i -offset):
            try:
                alt_color = (d[i,j] + d[i, j + 1] + d[i + 1, j])/3.
                hex_color = colormapper(alt_color, min_, max_)
                coordinates = triangle_coordinates(i,j, alt=True)
                output_file.write(svg_triangle(coordinates, hex_color))
            except KeyError:
                # Allow for some portions to have no color, such as the boundary
                pass
    output_file.write('</svg>\n')

# convert -density 1200 -resize 1000x1000 your.svg your.png
# convert -rotate 180 -density 1000 -resize 1000x1000 stationary_heatmap.svg stationary_heatmap.png

if __name__ == '__main__':
    N = 1200
    stationary_svg(N=N)
