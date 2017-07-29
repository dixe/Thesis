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

    res['total_acc'] = (1.0 * res['total_tp'] + res['total_tn']) / (res['total_tp'] + res['total_tn'] + res['total_fp'] + fn_i)

    res['total_precision'] = (res['total_tp'] * 1.0) / (res['total_tp'] + res['total_fp'])

    res['total_recall'] = (res['total_tp'] * 1.0) / (res['total_tp'] + res['total_fn'])

    res['total_f1'] = (2.0*res['total_precision'] * res['total_recall']) / (res['total_precision'] + res['total_recall'] )

    return res

def summerize(path):

    results = {}

    for r, ds, fs in os.walk(path):
        if f.endswith('results.csv'):
            guid = f.split('_')[0]
            settings = get_settings(guid)
            model_name = settings.model_name
            values = get_results(r+ "/" + f)

            folder = r.split('/')[-1]
            print folder
            continue
            results[guid] = calc_results(values)







if __name__ == "__main__":
    path = "/home/ltm741/thesis/datasets/final_test_sets/three_folder_test_set/"
