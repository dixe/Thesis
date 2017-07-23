import xml.etree.ElementTree as ET
import json
import os
import cv2
import pickle




def store(obj, name):
    with open(name,'w') as fp:
        pickle.dump(obj,fp)

def load(name):
    with open(name,'r') as fp:
        return pickle.load(fp)

def imshow(img, name = ""):
    cv2.imshow(name, img)
    cv2.waitKey()

def imshowExit(img,name = ""):
    imshow(img,name)
    exit()
