import sys, os
from sklearn.feature_extraction import image
import cv2

def generate_patches(load_path, img_width, img_height, num_patches):
    for r,ds,fs in  os.walk(load_path):
        for f in fs:
            img = cv2.imread(r + '//' + f)
            patches = image.extract_patches_2d(img,(img_width,img_height),max_patches=num_patches)

            for p in patches:
                yield p

if __name__ == "__main__":

    load_path = 'E://Speciale//auto_encoder_all_imgs//train//imgs'

    save_path = 'E://Speciale//auto_encoder_all_patches//train//imgs//'

    img_width = 64
    img_height = 64

    max_patches = 280/img_height *  176/img_width
    idd = 0
    for p in generate_patches(load_path, img_width, img_height, max_patches):
        save_name = "{0}{1}.png".format(save_path,idd)

        cv2.imwrite(save_name,p)
        idd +=1
