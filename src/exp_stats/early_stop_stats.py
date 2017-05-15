import sys
import numpy as np
import matplotlib.pyplot as plt

def plot_stop_vals(x, y, title):

    plt.plot(x,y)

    plt.xlabel("epochs")
    plt.ylabel("validation loss")

    plt.title = title

    plt.show()



def parse_stop_vals(vals_f):

    xs = []
    ys = []

    with open(vals_f,'r') as f:
        for line in f.readlines():
            x, y = line.split(',')
            x = float(x.split(':')[1])
            y = float(y.split(':')[1])
            xs.append(x)
            ys.append(y)

    res = np.array([xs,ys])

    return res


if __name__ == "__main__":

    f = sys.argv[1]

    xys = parse_stop_vals(f)

    plot_stop_vals(xys[0], xys[1], "GL early stopping values")
