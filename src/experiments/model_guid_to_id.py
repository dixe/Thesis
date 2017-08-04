import Weightstore as ws
import settings_stats as ss
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

id_dict[1] = ('simple_model', False)
id_dict[2] = ('simple_model', True)
id_dict[3] = ('simple_model_7_2_layer', True)
id_dict[4] = ('simple_model_7_5_5', True)
id_dict[5] = ('simple_model_7_fully_drop', True)
id_dict[6] = ('simple_model_7_nomax', True)
id_dict[7] = ('simple_model_min_7', True)
id_dict[8] = ('simple_model_min_7_drop', True)

dataset_ids = {} # global var, to avoid reading from disk too much

def get_model_id(guid_str):
    setting = ws.get_settings(guid_str)

    return id_dict[(setting.model_name, setting.sample_std)]


def get_dataset_id(name):
    global dataset_ids

    if not dataset_ids:

        with open('dataset_id.json','r') as f:
            dataset_ids = json.load(f)

    return dataset_ids[name]




def get_model_guid(model_id, dataset_id):

    dataset_ids = {}
    with open('dataset_id.json','r') as f:
        dataset_ids = json.load(f)

    dataset_names = {v: k for k, v in dataset_ids.items()}


    model_name, sample_std = id_dict[model_id]

    dataset_name = dataset_names[dataset_id]

    settings = ws.load_by_name_dataset_std(model_name, dataset_name, sample_std)

    settings_file_path = "settings_to_test.txt"
    settings_tested = list(map(lambda x : str(x.guid), ss.load_settings_from_file(settings_file_path)))


    for s in settings:
        if str(s.guid) in settings_tested:
            print(s.guid)


if __name__ == "__main__":

    if 'toguid' in sys.argv:
        model_id = int(sys.argv[-2])
        dataset_id = int(sys.argv[-1])

        get_model_guid(model_id, dataset_id)


        exit()

    guid_str = sys.argv[-1]

    idd = get_model_id(guid_str)

    print(idd)
