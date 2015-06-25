import csv
import sys
import numpy

from matplotlib import pyplot

def get_cmap(cmap_name=None):
    if not cmap_name:
        cmap = pyplot.get_cmap('jet')
    else:
        cmap = pyplot.get_cmap(cmap_name)
    return cmap    

def load_csv(filename):
    with open(filename) as handle:
        reader = csv.reader(handle)
        data = [row for row in reader]
    return data

def prepare_heatmap_data(data, xindex=0, yindex=1, cindex=-1, xfunc=float, yfunc=float, cfunc=float):
    # Grab horizontal and vertical coordinates.
    xs = list(sorted(set([xfunc(z[xindex]) for z in data])))
    ys = list(sorted(set([yfunc(z[yindex]) for z in data])))
    # Prepare to map to a grid.
    x_d = dict(zip(xs, range(len(xs))))
    y_d = dict(zip(ys, range(len(ys))))
    cs = numpy.zeros(shape=(len(ys), len(xs)))
    # Extract relevant data and populate color matrix, mapping to proper indicies.
    for row in data:
        x = xfunc(row[xindex])
        y = yfunc(row[yindex])
        c = cfunc(row[cindex])
        #cs[x_d[x]][y_d[y]] = c
        cs[y_d[y]][x_d[x]] = c
    return xs, ys, cs

def heatmap(xs, ys, cs, cmap=None, sep=10, offset=0., rounding=True, round_digits=3):
    if not cmap:
        cmap = get_cmap()
    plot_obj = pyplot.pcolor(cs, cmap=cmap)
    #plot_obj_2.callbacksSM.connect('changed', ImageFollower(color_obj_2))
    pyplot.colorbar()
    if rounding:
        xs = [round(x, round_digits) for x in xs]
        ys = [round(y, round_digits) for y in ys]
    
    #xticks = [round(x + offset, round_digits) for x in range(len(xs))]
    #yticks = [round(y + offset, round_digits) for y in range(len(ys))]
    xticks = [x + offset for x in range(len(xs))]
    yticks = [y + offset for y in range(len(ys))]

    pyplot.xticks(xticks[::sep], xs[::sep])
    pyplot.yticks(yticks[sep::sep], ys[sep::sep])

    return plot_obj    

def heatmap_ax(xs, ys, cs, cmap=None, sep=10, offset=0., rounding=True, round_digits=3, ax=None):
    if not cmap:
        cmap = get_cmap()
    if not ax:
        ax = pyplot.subplot()
    plot_obj = ax.pcolor(numpy.array(cs), cmap=cmap)
    #plot_obj_2.callbacksSM.connect('changed', ImageFollower(color_obj_2))
    fig = ax.get_figure()
    fig.colorbar(plot_obj)
    if rounding:
        xs = [round(x, round_digits) for x in xs]
        ys = [round(y, round_digits) for y in ys]
    
    #xticks = [round(x + offset, round_digits) for x in range(len(xs))]
    #yticks = [round(y + offset, round_digits) for y in range(len(ys))]
    xticks = [x + offset for x in range(len(xs))]
    yticks = [y + offset for y in range(len(ys))]

    ax.set_xticks(xticks[::sep], xs[::sep])
    ax.set_yticks(yticks[sep::sep], ys[sep::sep])

    return ax

def main(data=None, filename=None, xindex=0, yindex=1, cindex=-1, xfunc=float, yfunc=float, cfunc=float):
    if filename:
        data = load_csv(filename)
    if (not filename) and (not data):
        sys.stderr.write('Data or filename is required for heatmap.\n')
    xs, ys, cs = prepare_heatmap_data(data, xindex=xindex, yindex=yindex, cindex=cindex, xfunc=xfunc, yfunc=yfunc, cfunc=cfunc)
    heatmap(xs, ys, cs)
    
if __name__ == '__main__':
    filename = sys.argv[1]
    main(filename)
    pyplot.show()
    