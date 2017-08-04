import sys
import os
import numpy as np
import run_settings as rs
import Weightstore as ws
import cv2
import model_guid_to_id as mid
model_name_id = {}

model_name_id['simple_model'] = 1
model_name_id['simple_model'] = 2
model_name_id['simple_model_7_2_layer'] = 3
model_name_id['simple_model_7_5_5'] = 4
model_name_id['simple_model_7_fully_drop'] = 5
model_name_id['simple_model_7_nomax'] = 6
model_name_id['simple_model_min_7'] = 7
model_name_id['simple_model_min_7_drop'] = 8


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

    error = False

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
        res['total_acc'] = (1.0 * res['total_tp'] + res['total_tn']) / (res['total_tp'] + res['total_tn'] + res['total_fp'] + res['total_fn'])
    except:
        res['total_acc'] = 0
        error = True
    try:
        res['total_precision'] = (res['total_tp'] * 1.0) / (res['total_tp'] + res['total_fp'])
    except:
        res['total_precision'] = 0
        error = True
    try:
        res['total_recall'] = (res['total_tp'] * 1.0) / (res['total_tp'] + res['total_fn'])
    except:
        res['total_recall'] = 0
        error = True
    try:
        res['total_f1'] = (2.0*res['total_precision'] * res['total_recall']) / (res['total_precision'] + res['total_recall'] )
    except:
        res['total_f1'] = 0
        error = True
    return res, error


def table_row(res, model_id, dataset_id):


    return "{0} & {1} & {2:0.4f} & {3:0.4f} & {4:0.4f} & {5:0.4f} \\\\ \\hline \n".format(model_id,
                                                                                          dataset_id,
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
                std = settings.sample_std

                folder = r.split('/')[-1]


                values = get_results(r+ "/" + f)

                model_id = model_name_id[model_name]

                model_id = mid.get_model_id(guid)

                dataset_id = mid.get_dataset_id(settings.dataset)

                res, error = calc_results(values)

                if error:
                    print(guid)

                if not int(dataset_id) in [2,4,8]:
                    print(guid)


                results[folder][guid] = (res, model_id, dataset_id)


    for key in results.keys():
        header = "\\begin{tabular}{|c|c|c|c|c|c|} \\hline\n"
        header += "Model & Dataset & Acc & Recall & Precision & F1 \\\\ \\hline \n"
        body = ""
        footer = "\\end{tabular}"


        # sort guids by (model_id, dataset_id)
        guids = sorted(list(results[key].keys()), key=lambda x : (results[key][x][1],(results[key][x][2])))



        for guid in guids:
            res = results[key][guid]
            body += table_row(res[0], res[1], res[2])


        file_name = out_path + "/" + key.replace(' ', '_') + ".tex"

        print("Wirting: " + file_name)

        with open(file_name, 'w') as f:
            f.write(header + body + footer)











if __name__ == "__main__":

    path = "test_folder/"

    out_path = "final_exps/"

    summerize(path, out_path)
