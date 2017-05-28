import sys
import os
import numpy as np
import matplotlib.pyplot as plt

def plot_stop_vals(x, y, loss, title, y_label):

    print title, min(loss)
    plt.plot(x,y)
    y1 = [1 for i in range(len(x))]

    loss_min = min(loss)
    y2 = [loss_min for i in range(len(x))]


    plt.plot(x,loss)
    plt.plot(x,y1)
    plt.plot(x,y2)
    plt.xlabel("epochs")
    plt.ylabel(y_label)
    plt.xlim(0,len(x)-1)
    plt.ylim(0,max(y) + 0.1 * max(y))

    plt.title(title)

    plt.savefig("csvs_plots/" +title +".png")



def parse_stop_vals(vals_f):

    xs = []
    ys = []
    losses = []
    with open(vals_f,'r') as f:
        for line in f.readlines():
            splitted = line.split(',')
            x = splitted[0]
            y = splitted[1]
            loss = splitted[4] # val loss
            x = float(x.split(':')[1])
            y = float(y.split(':')[1])
            loss = float(loss.split(':')[1])
            xs.append(x)
            ys.append(y)
            losses.append(loss)

    res = np.array([xs,ys,losses])

    return res


def plot_many_csv(path):
    for r,ds,fs in os.walk(path):
        for f in fs:
            if f.endswith(".csv"):
                name = f.split('.')[0]
                xys = parse_stop_vals(r + '/' + f)
                y_label = name.split('_')[0]
                plot_stop_vals(xys[0],xys[1], xys[2], name, y_label)

if __name__ == "__main__":

    path = sys.argv[1]

    plot_many_csv(path)
    exit()


    xys = parse_stop_vals(f)

    plot_stop_vals(xys[0], xys[1], "GL early stopping values", "glt")
