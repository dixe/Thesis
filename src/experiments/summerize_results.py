import sys
import os
import numpy as np
import run_settings as rs
import Weightstore as ws
import cv2


def get_results(file_path):

    tp = []
    tn = []
    fp = []
    fn = []
    gen_time = []
    pred_time = []
    file_name = []
    with open(file_path, 'r') as f:
        for l in f.readlines():
            if l.startswith('tp'):
                continue
            vals = l.split(',')
            vals = list(map(lambda x : x.strip(), vals))
            tp.append(int(vals[0]))
            tn.append(int(vals[1]))
            fp.append(int(vals[2]))
            fn.append(int(vals[3]))

            gen_time.append(float(vals[4]))
            pred_time.append(float(vals[5]))
            file_name = vals[6]

    return (tp, tn, fp, fn, gen_time, pred_time, file_name)


def calc_results(values):

    tp = values[0]
    tn = values[1]
    fp = values[2]
    fn = values[3]

    res = {}

    res['total_tp'] = 0
    res['total_tn'] = 0
    res['total_fp'] = 0
    res['total_fn'] = 0

    res['avg_acc'] = 0


    for i in range(len(tp)):
        tp_i = tp[i]
        tn_i = tn[i]
        fp_i = fp[i]
        fn_i = fn[i]

        res['total_tp'] += tp_i
        res['total_tn'] += tn_i
        res['total_fp'] += fp_i
        res['total_fn'] += fn_i

        res['avg_acc'] += (1.0 * tp_i + tn_i) / (tp_i + tn_i + fp_i + fn_i)

    res['avg_acc'] /= 1.0 * len(tp)

    try:
        res['total_acc'] = (1.0 * res['total_tp'] + res['total_tn']) / (res['total_tp'] + res['total_tn'] + res['total_fp'] + fn_i)
    except:
        res['total_acc'] = 0

    try:
        res['total_precision'] = (res['total_tp'] * 1.0) / (res['total_tp'] + res['total_fp'])
    except:
        res['total_precision'] = 0

    try:
        res['total_recall'] = (res['total_tp'] * 1.0) / (res['total_tp'] + res['total_fn'])
    except:
        res['total_recall'] = 0

    try:
        res['total_f1'] = (2.0*res['total_precision'] * res['total_recall']) / (res['total_precision'] + res['total_recall'] )
    except:
        res['total_f1'] = 0
    return res


def table_row(res, model_name):

    return "{0} & {1} & {2} & {3} & {4}\\\\ \\hline".format(model_name,
                                                            res['total_acc'],
                                                            res['total_recall'],
                                                            res['total_precision'],
                                                            res['total_f1'])

def summerize(path, out_path):

    out_path = 'final_exps'
    results = {}

    for r, ds, fs in os.walk(path):
        for d in ds:
            results[d] = {}

    for r, ds, fs in os.walk(path):
        for f in fs:
            if f.endswith('results.csv'):
                guid = f.split('_')[0]
                settings = ws.get_settings(guid)
                model_name = settings.model_name

                folder = r.split('/')[-1]


                values = get_results(r+ "/" + f)
                results[folder][guid] = (calc_results(values),model_name)

    for key in results.keys():
        header = "\\begin{tabluar}{|c|c|c|c|c}\\\\ \\hline"
        header += "Model Name & Acc & Recall & Precision & F1 \\\\ \\hline"
        body = ""
        footer = "\\end{tabluar}"
        for guid in results[key]:
            res = results[res][guid]
            body +=table_row(res[0], res[1])


        file_name = ""

        print "Wirting: " + file_name

        with open(file_name, 'w') as f:
            w.rite(header + body + footer)











if __name__ == "__main__":
    path = "/home/ltm741/thesis/datasets/final_test_sets/three_folder_test_set/"
    summerize(path)
