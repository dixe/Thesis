import sys
import os
import numpy as np
import run_settings as rs
import simple_model as sm
import Weightstore as ws
import cv2
from keras.preprocessing.image import load_img, img_to_array


def predict_img(model, img, img_name, root, window_size = 64, stride = 2):

    strides_x = len(img[0][0]) / stride
    strides_y = len(img[0][0][0])/ stride

    preds = []

    #shape = patch.shape
    c = 0
    res_img = np.zeros((len(img[0][0]),len(img[0][0][0]),3))
    
    res_img[:,:,2] = img[0,0,:,:]
    res_img[:,:,1] = img[0,1,:,:]
    res_img[:,:,0] = img[0,2,:,:]



    c = 0
    for i in range(strides_x):
        for j in range(strides_y):
            
            patch = img[:,:,i*stride:i*stride + window_size, j*stride:j*stride+window_size]
            

            if patch.shape == (1,3,window_size,window_size):

                pred = model.predict(patch)
                if pred <= 0.5:
                    preds.append(pred)
                    #print preds[-1], i*stride, j*stride
                    res_img[i*stride + window_size/2, j*stride+window_size/2,:] = np.array([0,255,42])
                    #store_patch(patch, "{0}/patch_{1}_{2}.png".format(root,img_name,c))
                    c +=1
                

    cv2.imwrite("{0}/{1}".format(root, img_name), res_img)                    
    print sum(preds)






def store_patch(patch, name):

    res_patch = np.zeros((64,64,3))

    res_patch[:,:,2] = patch[0,0,:,:]
    res_patch[:,:,1] = patch[0,1,:,:]
    res_patch[:,:,0] = patch[0,2,:,:]

    cv2.imwrite(name,res_patch)





def get_settings_from_sysarg():
    sys.argv = filter(lambda x : x != '',sys.argv )
    guid_substring = sys.argv[-1]
    return ws.get_settings(guid_substring)


def test_simple(net):

    model = net.get_model_test()

    print rs.full_imgs_path
    for r,fs,fs in os.walk(rs.full_imgs_path):
        for f in fs:
            if f.endswith('impurities.bmp'):

                path = r + "/" + f
               
                print path
                
                img = np.array([img_to_array(load_img(path))])
                
                predict_img(model, img, f, r)


if __name__ == "__main__":

    settings = get_settings_from_sysarg()

    net = sm.get_net(settings)

    test_simple(net)
