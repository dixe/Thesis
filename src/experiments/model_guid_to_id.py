import Weightstore as ws
import sys


id_dict = {}
id_dict[('simple_model', False)] = 1
id_dict[('simple_model', True)] = 2
id_dict[('simple_model_7_2_layer', True)] = 3
id_dict[('simple_model_7_5_5', True)] = 4
id_dict[('simple_model_7_fully_drop', True)] = 5
id_dict[('simple_model_7_nomax', True)] = 6
id_dict[('simple_model_min_7', True)] = 7
id_dict[('simple_model_min_7_drop', True)] = 8




def get_id(guid_str):
    setting = ws.get_settings(guid_str)

    return id_dict[(setting.model_name, setting.sample_std)]


if __name__ == "__main__":

    guid_str = sys.argv[-1]

    idd = get_id(guid_str)

    print(idd)
