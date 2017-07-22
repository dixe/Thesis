import sys
import os
import cv2
import numpy as np


CLASS_TEST_PATH = "/home/nikolaj/Desktop/GQC Algorithm Test Output/"


def generate_class_bin_maps(base_path):

    for r,ds,fs in os.walk(base_path):
        for f in fs:
            if f.endswith("debug.bmp"):
                print f
                #print r + "/" + f

                img =  cv2.imread(r+ "/" + f)
                out_img = extract_class_bin_map(img)

                out_name = f.replace("debug.bmp","binmask.png")

                cv2.imwrite(r + "/" + out_name, out_img)


def extract_class_bin_map(img):



    mask = np.zeros((len(img),len(img[0]), 3),dtype=np.uint8)

    mask[img == [254,0,254]] = 255

    mask[img == [255,0,0]] = 0

    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

    mask[mask >0] = 255

    return mask




if __name__ == "__main__":


    if 'cbin' in sys.argv:
        generate_class_bin_maps(CLASS_TEST_PATH)

        exit()
