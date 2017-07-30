import run_settings as rs
import json

header = "\\begin{tabular}{|c|c|c|c|c|c|c|} \\hline\n"
header += "ataset id & \# training examples & \# validation examples & gamma & rotation & scaling & translation \\\\ \\hline\n"

body = ""
footer = "\\end{tabular}"

i = 1
name_to_id_map = {}
for s in rs.size_dict_train:
    if 'mini' in s or '32' in s:
        continue

    name_to_id_map[s] = i
    gm = 'gm' in s
    rot = 'rot' in s
    sc = 'sc' in s
    tl = 'tl' in s

    train_size = rs.size_dict_train[s]
    val_size = rs.size_dict_val[s]


    body += "{0} & {1} & {2} & {3} & {4} & {5} & {6} \\\\ \\hline\n".format(i, train_size,
                                                       val_size, gm,
                                                       rot,sc,tl)
    i+=1


with open('dataset_id.json','w') as f:
    json.dump(name_to_id_map,f)


print header + body + footer
