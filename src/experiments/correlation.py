import img_loader as IML
import cv2
import numpy as np
import sys
import os
import matplotlib.pyplot as plt
import json
from scipy.stats.stats import pearsonr, spearmanr


CLAAS_PATH = "E:/Speciale/CLAAS/GQC Algorithm Test Output/Masks/"

def rescale(x, new_min, new_max):
    old_min, old_max = np.min(x), np.max(x)

    return (new_max - new_min *1.0) / (old_max - old_min) * (x-old_min) + new_min



def make_cor_plot(data, guid, folder, cor = False):

    annos = data[:,0]
    pixels = data[:,1]

    if cor:
        return pearsonr(annos, pixels), spearmanr(annos, pixels)
    max_annos = np.max(annos)
    xs = [i for i in range(len(data))]

    pixels = rescale(pixels, 0 , max_annos)

    anno_plt, = plt.plot(xs,annos, label="Annotation")
    pixels_plt, = plt.plot(xs, pixels, label="Scaled #Pixel")
    plt.xlabel("Image sequence")
    plt.ylabel("#Annotation \ \n Scaled #Pixels")
    plt.title("Pixel to annotation correlation")

    plt.ylim(0, max_annos + 2) # we know that stopping is at 3

    lgd = plt.legend([anno_plt, pixels_plt], ["Annotation","Pixels"])

    plt.savefig("graphs/{0}_{1}_correlation_graph.png".format(guid,folder), bbox_extra_artists=(lgd,), bbox_inches='tight')
    plt.gcf().clear()



def get_all_cor_graphs(path, claas = False):


    for r, ds, fs in os.walk(path):
        folder = r.split('/')[-1]
        data = {}

        print folder
        xml_file = r + "/" + "Images Metadata Log.xml"
        if not os.path.exists(xml_file):
            continue
        i = 0

        if claas:
            data['claas'] = {}
            guid = 'claas'
            for f in fs:
                print "{0}/{1}".format(i, len(fs))
                i += 1
                if f.endswith('_ground_truth.bmp'):
                    frame = str(int(f.split('-')[0]))
                    frame_str = f.split('-')[0]

                    xml_p =  IML.XmlParser(xml_file, "all_impurities", frame)

                    bin_map = cv2.imread(CLAAS_PATH+ folder + "/" + "{0}-video_out_binmask.png".format(frame_str), 0)

                    num_annos = len(xml_p.get_annotations())

                    total_pixels = sum(bin_map.flatten() > 1)
                    data[guid][frame] = [num_annos, total_pixels]

            for guid in data:
                with open("graphs/{0}_{1}_data_dict.json".format(guid,folder), 'w') as f:
                    json.dump(data[guid], f)

            continue

        for f in fs:
            # Do all this for each guid, so write it all to a dict, and save, do graph after
            print "{0}/{1}".format(i, len(fs))
            i += 1

            if f.endswith('_output.png'):
                splited = f.split('_')

                guid = splited[0]

                if not guid in data:
                    data[guid] = {}


                frame = str(int(splited[1].split('-')[0]))
                xml_p =  IML.XmlParser(xml_file, "all_impurities", frame)

                bin_map = cv2.imread(r+'/' +f, 0)

                num_annos = len(xml_p.get_annotations())

                total_pixels = sum(bin_map.flatten() > 1)
                data[guid][frame] = [num_annos, total_pixels]


            for guid in data:
                with open("graphs/{0}_{1}_data_dict.json".format(guid,folder), 'w') as f:
                    json.dump(data[guid], f)


def plot_all_graphs(cor):

    for r, ds, fs in os.walk("graphs/"):
        for f in fs:
            data_dict = {}
            pcc_dict = {}
            if f.endswith('.json'):
                with open(r + "/" + f, 'r') as df:
                    data_dict = json.load(df)
                splitted = f.split('_')
                guid = splitted[0]
                folder = splitted[1]

                keys = sorted(data_dict.keys(), key = lambda x : int(x))


                data = []
                for key in keys:
                    data.append(data_dict[key])

                data = np.array(data)

                res = make_cor_plot(data,guid,folder, cor)

                if cor:
                    pearson, spearman = res
                    print "{0} & {1} & {2:0.6f} & {3:0.6f}".format(guid, folder, spearman[0], spearman[1])



if __name__ == "__main__":

    if 'plot' in sys.argv:
        plot_all_graphs('cor' in sys.argv)
        exit()

    claas = 'claas' in sys.argv
    if claas:
        path = "E:/Speciale/CLAAS/Datasets/Annotated/"

        get_all_cor_graphs(path, True)

    path = "/home/ltm741/thesis/datasets/final_test_sets/three_folder_test_set/"
    path = "E:/Speciale/CLAAS/Datasets/three_folder_test_set/"
    get_all_cor_graphs(path)
