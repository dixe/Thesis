import img_loader as IML
import os, os.path
import sys
import shutil
import cv2
import random

TRAIN_PATH = "E:/Speciale/CLAAS/Datasets/arg_data_sets/"
BASE_NAME = "patches_32{0}/"

def create_patches_for_path(path, settings):
    broken_patches = []
    non_broken_patches = []
    idb = 0
    idw = 0

    print "Saving to: " + TRAIN_PATH

    print path

    if not os.path.exists(TRAIN_PATH):
        os.mkdir(TRAIN_PATH)
    if not os.path.exists(TRAIN_PATH + "whole"):
        os.mkdir(TRAIN_PATH + "whole")

    if not os.path.exists(TRAIN_PATH + "broken"):
        os.mkdir(TRAIN_PATH + "broken")


    for r,ds,fs in os.walk(path):
        for f in fs:
            if f.endswith('.bmp'):
                print r + '/' + f
                imgLoader = IML.ImgLoad(f, r, settings)
                broken, whole = imgLoader.create_img_patches()

                for i in range(len(broken)):
                    name = "{0}.jpg".format(idb)
                    idb +=1
                    name = TRAIN_PATH + "broken/" + name
                    cv2.imwrite(name, broken[i])

                for i in range(len(whole)):
                    name = "{0}.jpg".format(idw)
                    idw +=1
                    name = TRAIN_PATH + "whole/" + name
                    cv2.imwrite(name, whole[i])




def test():
    img_path = "E:/Speciale/CLAAS/BG_Sequences_w_ROI_Annotated/November 7, 2014/"

    img_name = "00091-all_impurities.bmp"

    imgLoader = IML.ImgLoad(img_name,img_path)

    print imgLoader.create_img_patches()

    exit()


def shuffle_names():
    print "shuffling names"

    roots = [TRAIN_PATH + "broken/", TRAIN_PATH + "whole/"]
    for r in roots:

        count = num_files =len([f for f in os.listdir(r) if os.path.isfile(os.path.join(r,f))])

        names = range(count+1)

        random.shuffle(names)


        # remove tmp postfix
        for i in range(count+1):
            name = "{0}{1}.jpg".format(r,i)
            new_name = "{0}{1}_n.jpg".format(r, names[i])


            try:
                os.rename(name,new_name)
            except:
                print name, new_name



def path_sub_name(rot,sc,tl,gm):
    return "{0}{1}{2}{3}".format("_rot" if rot else "", "_sc" if sc else "", "_tl" if tl else "", "_gm" if gm else "")


def create_all_patches_comb(anno_path, base_save_path, names = ['rot','sc','tl','gm']):
    rot_tf = [False, True] if 'rot' in names else [False]
    sc_tf = [False, True] if 'sc' in names else [False]
    tl_tf = [False, True] if 'tl' in names else [False]
    gm_tf = [False, True] if 'gm' in names else [False]

    print "Creating for {0}".format(names)
    global TRAIN_PATH
    for rot in rot_tf:
        for sc in sc_tf:
            for tl in tl_tf:
                for gm in gm_tf:

                    TRAIN_PATH = base_save_path + BASE_NAME.format(path_sub_name(rot,sc,tl,gm))

                    if os.path.isdir(TRAIN_PATH):
                        continue


                    settings = IML.Settings(rot, sc, tl, gm)

                    create_patches_for_path(anno_path, settings)

def shuffle_all_comb(base_save_path, names = ['rot','sc','tl','gm']):
    rot_tf = [False, True] if 'rot' in names else [False]
    sc_tf = [False, True] if 'sc' in names else [False]
    tl_tf = [False, True] if 'tl' in names else [False]
    gm_tf = [False, True] if 'gm' in names else [False]

    global TRAIN_PATH
    for rot in rot_tf:
        for sc in sc_tf:
            for tl in tl_tf:
                for gm in gm_tf:

                    TRAIN_PATH = base_save_path + BASE_NAME.format(path_sub_name(rot,sc,tl,gm))

                    shuffle_names()


if __name__ == "__main__":
    anno_path = "E:/Speciale/CLAAS/BG_Sequences_w_ROI_Annotated/"

    base_save_path = "E:/Speciale/CLAAS/Datasets/arg_data_sets_few_whole_32/"
    rotation = False
    scale = False
    translate = False
    gamma = True
    settings = IML.Settings(rotation, scale, translate, gamma)
    data_args = ['sc','gm']

    if 's' in sys.argv:
        if 'c' in sys.argv:
            create_patches_for_path(anno_path, settings)
        shuffle_all_comb(base_save_path, names = data_args)
        exit()

    if 'a' in sys.argv:
        #base_save_path = sys.argv[sys.argv.index('p') + 1]

        create_all_patches_comb(anno_path, base_save_path, names=data_args)
    else:
        create_patches_for_path(anno_path, settings)
