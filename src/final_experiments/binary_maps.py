import sys
import os
import cv2
import numpy as np
import shutil
import img_loader as IML

CLASS_TEST_PATH = "E:/Speciale/CLAAS/GQC Algorithm Test Output/"

CLAAS_MASK_OUTPUT = "E:/Speciale/CLAAS/GQC Algorithm Test Output/Masks"

DIKU_TEST_SET = 'E:/Speciale/CLAAS/DIKU Test dataset/'

GROUND_TRUTH_PATH = 'E:/Speciale/CLAAS/DIKU Test dataset/ground_truth/'

CLAAS_LOG_FILE = "Images Metadata Log.xml"





def generate_class_bin_maps(base_path):

    for r,ds,fs in os.walk(base_path):
        for f in fs:
            if f.endswith("debug.bmp"):
                print f
                #print r + "/" + f

                img =  cv2.imread(r+ "/" + f)
                out_img = extract_class_bin_map(img)

                d = r.split('/')[-1]

                out_name = f.replace("debug.bmp","binmask.png")
                root = CLAAS_MASK_OUTPUT + "/" + d + "/"

                if not os.path.exists(root):
                    os.makedirs(root)
                cv2.imwrite(root + out_name, out_img)

                if not os.path.exists(root + CLAAS_LOG_FILE):
                    shutil.copy(r + "/" + CLAAS_LOG_FILE,root + CLAAS_LOG_FILE)




def extract_class_bin_map(img):

    mask = np.zeros((len(img),len(img[0]),1),dtype=np.uint8)

    maMask = img[:,:,0] == 254
    maMask &= img[:,:,1] == 0
    maMask &= img[:,:,2] == 254

    mask[maMask] = 255

    return mask





def create_ground_truth(path):

    for r,ds,fs in os.walk(path):
        for f in fs:
            if f.endswith("all_impurities.bmp"):
                print f

                ground_img = np.zeros((376.240,1),dtype=np.uint8)

                settings = IML.Settings(False,False, False, False)

                imgLoader = IML.ImgLoad(f, r, settings)



                for a in imgLoader.annotations:
                    print a.center, a.radius
                    cv2.circle(ground_img,a.center, a.radius, 255, -1)

                cv2.imwrite(





if __name__ == "__main__":

    if 'gt' in sys.argv:
        create_ground_truth(DIKU_TEST_SET)
        exit()



    if 'cbin' in sys.argv:
        generate_class_bin_maps(CLASS_TEST_PATH)

        exit()