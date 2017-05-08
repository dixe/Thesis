import img_loader as IML
import os, os.path
import sys
import cv2
import random
import shutil


DATA_PATH = "E:/Speciale/CLAAS/Datasets/patches-64/"
OUT_PATH = "E:/Speciale/CLAAS/Datasets/Training_Validation_datasets/data-64-X-Y/"


def create_train_val(data_path, out_path, split = 0.15):

    if not os.path.exists(OUT_PATH):
        os.mkdir(OUT_PATH)
    if not os.path.exists(OUT_PATH + "train"):
        os.mkdir(OUT_PATH + "train")
        if not os.path.exists(OUT_PATH + "train/whole"):
            os.mkdir(OUT_PATH + "train/whole")
        if not os.path.exists(OUT_PATH + "train/broken"):
            os.mkdir(OUT_PATH + "train/broken")

    if not os.path.exists(OUT_PATH + "validation"):
        os.mkdir(OUT_PATH + "validation")
        if not os.path.exists(OUT_PATH + "validation/whole"):
            os.mkdir(OUT_PATH + "validation/whole")
        if not os.path.exists(OUT_PATH + "validation/broken"):
            os.mkdir(OUT_PATH + "validation/broken")





    # head count of broken and valid
    data_p_broken = data_path + "broken"
    data_p_whole = data_path + "whole"

    count_broken = num_files =len([f for f in os.listdir(data_p_broken) if os.path.isfile(os.path.join(data_p_broken,f))])
    count_whole = num_files =len([f for f in os.listdir(data_p_whole) if os.path.isfile(os.path.join(data_p_whole,f))])


    print count_whole, count_broken


    broken_train_end = int(count_broken * (1-split))
    whole_train_end = int(count_whole * (1-split))

    print "broken in train: {0}, whole in train: {1}".format(broken_train_end, whole_train_end)


    for c in range(broken_train_end):
        shutil.copy("{0}broken/{1}.jpg".format(data_path,c), "{0}train/broken/{1}.jpg".format(out_path,c))

    for c in range(broken_train_end,count_broken):
        shutil.copy("{0}broken/{1}.jpg".format(data_path,c), "{0}validation/broken/{1}.jpg".format(out_path,c))


    for c in range(whole_train_end):
        shutil.copy("{0}whole/{1}.jpg".format(data_path,c), "{0}train/whole/{1}.jpg".format(out_path,c))

    for c in range(whole_train_end,count_whole):
        shutil.copy("{0}whole/{1}.jpg".format(data_path,c), "{0}validation/whole/{1}.jpg".format(out_path,c))




def test():
    img_path = "E:/Speciale/CLAAS/BG_Sequences_w_ROI_Annotated/November 7, 2014/"

    img_name = "00091-all_impurities.bmp"

    imgLoader = IML.ImgLoad(img_name,img_path)

    print imgLoader.create_img_patches()

    exit()




if __name__ == "__main__":

    create_train_val(DATA_PATH, OUT_PATH)
