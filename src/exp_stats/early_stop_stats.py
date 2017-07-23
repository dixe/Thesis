import sys
import os
import numpy as np
import matplotlib.pyplot as plt

def plot_stop_vals(x, pqt, train_loss, train_acc, val_loss, val_acc, title, y_label, path):


    x = map(int,x)

    print title, min(val_loss)
    pqt_plt, = plt.plot(x,pqt, label="pqt")

    loss_min = min(val_loss)
    min_val_loss = [loss_min for i in range(len(x))]


    train_loss_plt, = plt.plot(x, train_loss, label="train loss")
    train_acc_plt, =plt.plot(x, train_acc, label="train acc")
    val_loss_plt, = plt.plot(x, val_loss, label="val loss")
    val_acc_plt, = plt.plot(x, val_acc, label="val acc")
    min_val_loss_plt, = plt.plot(x, min_val_loss, label="min val loss")

    plt.xlabel("epochs")
    plt.ylabel(y_label)
    plt.xlim(0,len(x))
    plt.ylim(0,3.1) # we know that stopping is at 3

    plt.title(title)

    lgd = plt.legend([pqt_plt, train_loss_plt, train_acc_plt, val_loss_plt, val_acc_plt, min_val_loss_plt],["pqt", "train loss", "train acc", "val loss", "val_acc", "min val loss"], bbox_to_anchor=(1.04,1), loc=2)

    plt.savefig(path + "/" +title +".png", bbox_extra_artists=(lgd,), bbox_inches='tight')
    plt.gcf().clear()

'''
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
'''
def parse_stop_vals(vals_f):

    res = []
    with open(vals_f,'r') as f:
        for line in f.readlines():
            splitted = line.split(',')
            epoch = splitted[0]
            pqt = splitted[1]
            tl = splitted[2]
            ta = splitted[3]
            vl = splitted[4]
            va = splitted[5]

            row = [epoch, pqt, tl, ta, vl, va]
            row = map(lambda r: float(r.split(':')[1]),row)
            res.append(row)
    res = np.array(res)
    return res


def plot_many_csv(path):
    for r,ds,fs in os.walk(path):
        for f in fs:
            if f.endswith(".csv"):
                name = f.split('.')[0]
                vals = parse_stop_vals(r + '/' + f)
                y_label = name.split('_')[0]
                plot_stop_vals(vals[:,0], vals[:,1], vals[:,2], vals[:,3], vals[:,4],vals[:,5], name, y_label, path)

if __name__ == "__main__":

    path = sys.argv[1]

    plot_many_csv(path)
    exit()


    vals = parse_stop_vals(f)

    plot_stop_vals(vals[0], vals[1], "GL early stopping values", "glt")
