import sys
import os
import cv2
import numpy as np
import img_loader as IML





def found_annos(path, claas_path = None):


    claas = claas_path is not None

    anno_img = np.zeros((240,376))

    for r, ds, fs in os.walk(path):
        xml_file = r + "/" + "Images Metadata Log.xml"
        if not os.path.exists(xml_file):
            continue

        total_annos = {}
        found = {}

        if claas:
            total_annos['claas'] = 0
            found['claas'] = 0

        files = len(fs)
        i = 0
        for f in fs:
            print "{0}/{1}".format(i, files)
            i += 1
            if claas:
                if f.endswith('_ground_truth.bmp'):
                    frame = str(int(f.split('-')[0]))
                    xml_p =  IML.XmlParser(xml_file, "all_impurities", frame)

                    xml_p =  IML.XmlParser(xml_file, "all_impurities", frame)

                    bin_map = cv2.imread(r+'/' +f, 0)

                    annos = xml_p.get_annotations()

                    total_annos['claas'] += len(annos)

                    for a in annos:
                        cv2.circle(anno_img, a.center, a.radius, 255, -1)

                        found['claas'] += 1 if sum(np.logical_and(anno_img.flatten(),bin_map.flatten())) > 1 else 0

                        anno_img = np.zeros((240,376))
            else:
                if f.endswith('_output.png'):
                    splited = f.split('_')
                    guid = splited[0]

                    if not guid in total_annos:
                        total_annos[guid] = 0
                        found[guid] = 0

                    frame = str(int(splited[1].split('-')[0]))
                    xml_p =  IML.XmlParser(xml_file, "all_impurities", frame)

                    bin_map = cv2.imread(r+'/' +f, 0)

                    annos = xml_p.get_annotations()

                    total_annos[guid] += len(annos)

                    for a in annos:
                        cv2.circle(anno_img, a.center, a.radius, 255, -1)

                        found[guid] += 1 if sum(np.logical_and(anno_img.flatten(),bin_map.flatten())) > 1 else 0

                        anno_img = np.zeros((240,376))

        with open('num_found_results.csv', 'w') as outf:
            outf.write('guid, found, total\n')
            for guid in total_annos:
                outf.write("{0}, {1}, {2}\n".format(guid, found[guid], total_annos[guid]))

if __name__ == "__main__":

    path = "test_folder"
    if 'claas' in sys.argv:
        path = "E:/Speciale/CLAAS/DIKU Test dataset/Annotated/"
        claas_path = "E:/Speciale/CLAAS/GQC Algorithm Test Output/"
        found_annos(path, claas_path)
    else:
        found_annos(path)
