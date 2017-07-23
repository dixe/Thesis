import sys
import os
import numpy as np
import matplotlib.pyplot as plt

def table_stop_values(name, train_loss, train_acc, val_loss, val_acc):
    """
    return latex string for this row
    """

    min_val_loss = min(val_loss)

    row  = np.where(val_loss == min_val_loss)

    val_acc = val_acc[row][0]
    train_loss = train_loss[row][0]
    train_acc = train_acc[row][0]
    val_loss = val_loss[row][0]


    row_string = "{0} & {1:.4f} & {2:.4f} & {3:.4f} & {4:.4f}\\\\ \\hline\n".format(name.replace("_","\\_"),train_loss, train_acc, val_loss, val_acc)

    return row_string



def table_test_results(file_name):
    vals_dict = get_val_results(file_name)


    rows_string = ""

    header_string = """\\begin{tabular}{c|c|c|c|c|c}
DATASET &  VAL\_LOSS & VAL\_ACC & RECALL & PRECISION & F1 \\\\ \hline\n"""

    footer_string = """\end{tabular}"""


    for name in vals_dict:
        vals = vals_dict[name]
        pres = vals[3]
        rec = vals[2]
        f1 = 2.0 * (pres * rec)/(pres+ rec)

        rows_string += "{0} & {1:.4f} & {2:.4f} & {3:.4f} & {4:.4f} & {5:.4f}\\\\ \\hline\n".format(name.replace("_","\\_"), vals[0], vals[1], rec, pres, f1)


    return header_string + rows_string + footer_string

def parse_stop_vals(vals_f):

    res = []
    with open(vals_f,'r') as f:
        for line in f.readlines():
            splitted = line.split(',')
            tl = splitted[2]
            ta = splitted[3]
            vl = splitted[4]
            va = splitted[5]

            row = [tl,ta,vl,va]
            row = map(lambda r: float(r.split(':')[1]),row)
            res.append(row)
    res = np.array(res)
    return res


def table_many_csv(path):
    """
    Make a table of the best validation values for each dataset (csv file)
    """
    rows_string = ""
    header_string = """\\begin{tabular}{c|c|c|c|c}
DATASET & TRAIN\_ACC & TRIAN\_LOSS & VAL\_LOSS & VAL\_ACC \\\\ \hline\n"""

    footer_string = """\end{tabular}"""

    for r,ds,fs in os.walk(path):
        for f in fs:
            if f.endswith(".csv"):
                name = f.split('.')[0]
                name = name.replace("pqt_","")
                vals = parse_stop_vals(r + '/' + f)
                rows_string += table_stop_values(name, vals[:,0], vals[:,1], vals[:,2], vals[:,3])

    return header_string + rows_string + footer_string


def get_val_results(file_name):

    with open(file_name,'r') as f:
        lines = f.readlines()

    vals_dict = {}
    for l in lines:
        if 'acc' not in l:
            name, loss, acc, recall, precision = l.split(',')

            vals_dict[name.strip()] = (float(loss), float(acc), float(recall), float(precision))

    return vals_dict



if __name__ == "__main__":

    # file_name = "eval_all_simple_model.txt"

    #table_string = table_test_results(file_name)
    #print table_string

    #exit()

    path = sys.argv[1]

    table = table_many_csv(path)

    print table
