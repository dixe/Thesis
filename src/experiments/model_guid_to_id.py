import Weightstore as ws
import sys
import json

id_dict = {}
id_dict[('simple_model', False)] = 1
id_dict[('simple_model', True)] = 2
id_dict[('simple_model_7_2_layer', True)] = 3
id_dict[('simple_model_7_5_5', True)] = 4
id_dict[('simple_model_7_fully_drop', True)] = 5
id_dict[('simple_model_7_nomax', True)] = 6
id_dict[('simple_model_min_7', True)] = 7
id_dict[('simple_model_min_7_drop', True)] = 8

dataset_ids = {}

def get_model_id(guid_str):
    setting = ws.get_settings(guid_str)

    return id_dict[(setting.model_name, setting.sample_std)]


def get_dataset_id(name):
    global dataset_ids

    if not dataset_ids:

        with open('dataset_id.json','r') as f:
            dataset_ids = json.load(f)

    return dataset_ids[name]


def get_model_guid(dataset_id, model_id)



if __name__ == "__main__":

    if 'toguid' in sys.argv:
        dataset_id = sys.arv[-2]
        model_id = sys.argv[-1]



        exit()

    guid_str = sys.argv[-1]

    idd = get_model_id(guid_str)

    print(idd)
