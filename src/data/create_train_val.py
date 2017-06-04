import img_loader as IML
import os, os.path
import sys
import cv2
import random
import shutil

BASE_NAME = "patches_32{0}/"

def create_train_val(data_path, out_path, split = 0.15):

    if not os.path.exists(out_path):
        os.mkdir(out_path)
    if not os.path.exists(out_path + "train"):
        os.mkdir(out_path + "train")
        if not os.path.exists(out_path + "train/whole"):
            os.mkdir(out_path + "train/whole")
        if not os.path.exists(out_path + "train/broken"):
            os.mkdir(out_path + "train/broken")

    if not os.path.exists(out_path + "validation"):
        os.mkdir(out_path + "validation")
        if not os.path.exists(out_path + "validation/whole"):
            os.mkdir(out_path + "validation/whole")
        if not os.path.exists(out_path + "validation/broken"):
            os.mkdir(out_path + "validation/broken")





    # head count of broken and valid
    data_p_broken = data_path + "broken"
    data_p_whole = data_path + "whole"

    count_broken = num_files =len([f for f in os.listdir(data_p_broken) if os.path.isfile(os.path.join(data_p_broken,f))])
    count_whole = num_files =len([f for f in os.listdir(data_p_whole) if os.path.isfile(os.path.join(data_p_whole,f))])


    print count_broken, count_whole


    broken_train_end = int(count_broken * (1-split))
    whole_train_end = int(count_whole * (1-split))


    print ""
    print "Starting broken"

    missing_train = 0
    missing_val = 0

    for c in range(broken_train_end):
        try:
            shutil.copy("{0}broken/{1}_n.jpg".format(data_path,c), "{0}train/broken/{1}.jpg".format(out_path,c))
        except IOError :
            missing_train += 1
            print "TRAIN not found: broken/{0}_n.jpg".format(c)


    for c in range(broken_train_end,count_broken):
        try:
            shutil.copy("{0}broken/{1}_n.jpg".format(data_path,c), "{0}validation/broken/{1}.jpg".format(out_path,c))
        except IOError :
            missing_val += 1
            print "VAL not found: broken/{0}_n.jpg".format(c)


    print ""
    print "Starting whole"
    for c in range(whole_train_end):
        try:
            shutil.copy("{0}whole/{1}_n.jpg".format(data_path,c), "{0}train/whole/{1}.jpg".format(out_path,c))
        except IOError :
            missing_train += 1
            print "TRAIN not found: whole/{0}_n.jpg".format(c)

    for c in range(whole_train_end,count_whole):
        try:
            shutil.copy("{0}whole/{1}_n.jpg".format(data_path,c), "{0}validation/whole/{1}.jpg".format(out_path,c))
        except IOError :
            missing_val += 1
            print "VAL not found : whole/{0}_n.jpg".format(c)

    print "broken in train: {0}, whole in train: {1}, total train: {2}".format(broken_train_end, whole_train_end, broken_train_end + whole_train_end - missing_train)
    print "broken in val: {0}, whole in val: {1}, total val: {2}".format(count_broken -broken_train_end, count_whole - whole_train_end, count_broken -broken_train_end + count_whole - whole_train_end - missing_val)


def test():
    img_path = "E:/Speciale/CLAAS/BG_Sequences_w_ROI_Annotated/November 7, 2014/"

    img_name = "00091-all_impurities.bmp"

    imgLoader = IML.ImgLoad(img_name,img_path)

    print imgLoader.create_img_patches()

    exit()



def path_sub_name(rot,sc,tl,gm):
    return "{0}{1}{2}{3}".format("_rot" if rot else "", "_sc" if sc else "", "_tl" if tl else "", "_gm" if gm else "")


def train_val_all_comb(names = ['rot','sc','tl','gm']):
    rot_tf = [False, True] if 'rot' in names else [False]
    sc_tf = [False, True] if 'sc' in names else [False]
    tl_tf = [False, True] if 'tl' in names else [False]
    gm_tf = [False, True] if 'gm' in names else [False]

    for rot in rot_tf:
        for sc in sc_tf:
            for tl in tl_tf:
                for gm in gm_tf:

                    dataset_name = BASE_NAME.format(path_sub_name(rot,sc,tl,gm))
                    print ""
                    print ""
                    print dataset_name

                    data_path = "E:/Speciale/CLAAS/Datasets/arg_data_sets_few_whole_32/{0}/".format(dataset_name)
                    out_path = "E:/Speciale/CLAAS/Datasets/arg_data_sets_few_whole_32/train_val/{0}/".format(dataset_name)


                    create_train_val(data_path, out_path)

if __name__ == "__main__":

    train_val_all_comb(['sc','gm'])
