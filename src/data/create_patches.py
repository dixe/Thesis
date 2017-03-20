import img_loader as IML
import os
import cv2


TRAIN_PATH = "/home/nikolaj/CLAAS/Datasets/patches-64/"

def create_patches_for_path(path):
    broken_patches = []
    non_broken_patches = []
    idb = 0
    idw = 0
    print path
    for r,ds,fs in os.walk(path):

        for f in fs:
            if f.endswith('all_impurities.bmp'):
                print r + '/' + f
                imgLoader = IML.ImgLoad(f, r)
                broken, whole = imgLoader.create_img_patches()

                print len(broken), len(whole)
                for i in range(len(broken)):
                    name = "{3}-{0:04}-{1}-{2}.jpg".format(int(imgLoader.frame),imgLoader.img_name,i,idb)
                    idb +=1
                    name = TRAIN_PATH + "broken/" + name
                    cv2.imwrite(name, broken[i])

                for i in range(len(whole)):
                    name = "{3}-{0:04}-{1}-{2}.jpg".format(int(imgLoader.frame),imgLoader.img_name,i,idw)
                    idw +=1
                    name = TRAIN_PATH + "whole/" + name
                    cv2.imwrite(name, whole[i])




def test():
    img_path = "~/CLAAS/BG_Sequences_w_ROI_Annotated/November 7, 2014/"

    img_name = "00091-all_impurities.bmp"

    imgLoader = IML.ImgLoad(img_name,img_path)

    print imgLoader.create_img_patches()

    exit()



def create_patches_for_path_new(path):
    broken_patches = []
    non_broken_patches = []
    idb = 0
    idw = 0
    print path
    for r,ds,fs in os.walk(path):

        for f in fs:
            if f.endswith('all_impurities.bmp'):
                print r + '/' + f
                imgLoader = IML.ImgLoad(f, r)
                broken, whole = imgLoader.create_img_patches()

                print len(broken), len(whole)
                for i in range(len(broken)):
                    name = "{3}-{0:04}-{1}-{2}.jpg".format(int(imgLoader.frame),imgLoader.img_name,i,idb)
                    idb +=1
                    name = TRAIN_PATH + "broken/" + name
                    cv2.imwrite(name, broken[i])

                for i in range(len(whole)):
                    name = "{3}-{0:04}-{1}-{2}.jpg".format(int(imgLoader.frame),imgLoader.img_name,i,idw)
                    idw +=1
                    name = TRAIN_PATH + "whole/" + name
                    cv2.imwrite(name, whole[i])




def test():
    img_path = "~/CLAAS/BG_Sequences_w_ROI_Annotated/November 7, 2014/"

    img_name = "00091-all_impurities.bmp"

    imgLoader = IML.ImgLoad(img_name,img_path)

    print imgLoader.create_img_patches()

    exit()


if __name__ == "__main__":

    anno_path = "/home/nikolaj/CLAAS/GQC_Maize_BG_Annotation/"
    create_patches_for_path(anno_path)
