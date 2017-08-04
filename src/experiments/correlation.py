import img_loader as IML
import cv2
import numpy as np
import sys
import os
import matplotlib.pyplot as plt



def rescale(x, new_min, new_max):
    old_min, old_max = np.min(x), np.max(x)

    return (new_max - new_min *1.0) / (old_max - old_min) * (x-old_min) + new_min



def make_cor_plot(data):
    annos = data[:,0]
    pixels = data[:,1]

    max_annos = np.max(annos)
    xs = [i for i in range(len(data))]

    pixels = rescale(pixels, 0 , max_annos)



    anno_plt, = plt.plot(xs,annos, label="Annotation")
    pixels_plt, = plt.plot(xs, pixels, label="Scaled #Pixel")
    plt.xlabel("Image sequence")
    plt.ylabel("#Annotation  \\n Scaled #Pixels")
    plt.title("Pixel to annotation correlation")

    lgd = plt.legend([anno_plt, pixels_plt], ["Annotation","Pixels"], bbox_to_anchor=(1.04,1), loc=2)

    return lgd




def get_all_cor_graphs(path):

    for r, ds, fs in os.walk(path):

        data = []

        xml_file = r + "/" + "Images Metadata Log.xml"
        if not os.path.exists(xml_file):
            continue
        for f in fs:
            if f.endswith('_output.png'):
                splited = f.split('_')

                frame = str(int(splited[1].split('-')[0]))
                xml_p =  IML.XmlParser(xml_file, "all_impurities", frame)

                bin_map = cv2.imread(r+'/' +f, 0)

                num_annos = len(xml_p.get_annotations())

                total_pixels = sum(bin_map.flatten() > 1)
                data.append([num_annos, total_pixels])

        lgd = make_cor_plot(np.array(data))

        folder = r.split('/')[-1]

        plt.savefig("graphs/{0}_correlation_graph.png", bbox_extra_artists=(lgd,), bbox_inches='tight')
        plt.gcf().clear()



if __name__ == "__main__":
    path = "/home/ltm741/thesis/datasets/final_test_sets/three_folder_test_set/"
    path = "test_folder"
    get_all_cor_graphs(path)
