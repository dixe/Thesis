import cv2
import sys
import numpy as np
from keras.preprocessing.image import load_img, img_to_array

def create_img_patches(img, window_size, stride):

    strides_x = len(img[0][0]) / stride
    strides_y = len(img[0][0][0])/ stride


    patches = []

    for i in range(strides_x):
        for j in range(strides_y):
            if not in_roi(i,j,stride, window_size):
                continue

            patch = img[:,:,i*stride:i*stride + window_size, j*stride:j*stride+window_size]

            if patch.shape == (1,3,window_size,window_size):
                patches.append(patch)

    patches = np.array(patches)

    return patches

def in_roi(i,j,s,w):
    y = i*s
    x = j*s
    ym = i*s +w
    xm = j*s +w

    return y >= 20 and ym <= 200



def get_arg_from_sysargv(arg_name):
    if arg_name not in sys.argv:
        return Non

    index = sys.argv.index(arg_name)
    return sys.argv[index + 1]

if __name__ == "__main__":

    img_name = get_arg_from_sysargv('img')
    img = np.array([img_to_array(load_img(img_name))])

    print "Start create"
    
    patches = create_img_patches(img, 64,1)

    print "Finish Create"

    print len(patches)
