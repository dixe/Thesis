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

    ground_img = None

    imgLoader = None

    for r,ds,fs in os.walk(path):
        for f in fs:
            d = r.split('/')[-1]

            if f.endswith("all_impurities.bmp") and "ground_truth" not in d:

                print f

                root = GROUND_TRUTH_PATH + "/" + d + "/"

                out_path = root + f.replace(".bmp", '_ground_truth.bmp')
                if os.path.exists(out_path):
                    print "bla"
                    continue

                ground_img = np.zeros((240, 376), dtype=np.uint8)


                settings = IML.Settings(False,False, False, False)

                imgLoader = IML.ImgLoad(f, r, settings)


                if not os.path.exists(root):
                    os.makedirs(root)

                for a in imgLoader.annotations:
                    cv2.circle(ground_img, a.center, a.radius, 255, -1)

                mask =  np.zeros((240, 376), dtype=np.uint8)

                x_min, y_min, x_max, y_min = imgLoader.get_roi_uncorrected()

                mask[x_min:x_max, y_min: y_max] = 255


                cv2.imshow("ground", ground_img)

                cv2.imshow("masked", np.logical_and(mask, ground_img))

                cv2.waitKey()


                #cv2.imwrite(out_path, ground_img)







if __name__ == "__main__":

    if 'gt' in sys.argv:
        create_ground_truth(DIKU_TEST_SET)
        exit()



    if 'cbin' in sys.argv:
        generate_class_bin_maps(CLASS_TEST_PATH)

        exit()
