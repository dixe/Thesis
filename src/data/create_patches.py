import img_loader as IML
import os, os.path
import sys
import cv2
import random

TRAIN_PATH = "E:/Speciale/CLAAS/Datasets/patches-64/"

def create_patches_for_path(path):
    broken_patches = []
    non_broken_patches = []
    idb = 0
    idw = 0
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
                imgLoader = IML.ImgLoad(f, r)
                broken, whole = imgLoader.create_img_patches()

                print len(broken), len(whole)
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

    roots = [TRAIN_PATH + "broken/",TRAIN_PATH + "whole/"]
    for r in roots:



        count = num_files =len([f for f in os.listdir(r) if os.path.isfile(os.path.join(r,f))])

        names = range(count+1)

        random.shuffle(names)



        # remove tmp postfix
        for i in range(count+1):
            name = "{0}{1}_n.jpg".format(r,i)
            new_name = "{0}{1}.jpg".format(r,i)

            try:
                os.rename(name,new_name)
            except:
                print name
                print new_name



if __name__ == "__main__":
    anno_path = "E:/Speciale/CLAAS/BG_Sequences_w_ROI_Annotated/"


    if 's' in sys.argv:
        if 'c' in sys.argv:
            create_patches_for_path(anno_path)
        shuffle_names()
    else:
        create_patches_for_path(anno_path)
