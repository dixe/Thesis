import cv2
import xml.etree.ElementTree as ET
import os.path
import json
import utils as UT
import random as rand
import numpy as np
import sys

IMG_META_XML_NAME = "Images Metadata Log.xml"

PATCH_SIZE = 32
PATCH_SIZE_HALF = PATCH_SIZE / 2
OFFSET_RANGE = 5
NUM_NON_BROKEN_PATHCES = 0


class ImgLoad(object):

    def __init__(self, name, path, settings):
        self.name = name
        self.frame, self.img_name = self.get_frame_name(name)
        self.path = path
        self.settings = settings

        self.XmlParser = XmlParser(self.path  +'/' + IMG_META_XML_NAME, self.img_name, self.frame)
        self.valid = self.XmlParser.valid

        self.annotations = self.XmlParser.get_annotations()

        self.roi = self.get_roi()

        self.image = None

    def get_frame_name(self, img_name):
        frame, name = img_name.split('-')

        name = name.split('.')[0]
        frame = str(int(frame))


        return frame, name


    def draw_broken(self):
        for anno in self.annotations:
            cv2.circle(self.img(), anno.center, anno.size, (0,0,225))

    def visualize_image(self):

        self.draw_broken()
        self.show_img(self.img())


    def show_img(self, img):
        cv2.imshow("",img)
        cv2.waitKey()

    def img(self):

        if self.image == None:
            self.image = cv2.imread(self.path + '/' + self.name)


        return self.image

    def create_img_patches(self):
        if not self.valid:
            return [],[]
        broken_patches, real_annos = self.create_broken_patches()

        non_broken_patches = self.create_non_broken_patches(real_annos)


        return broken_patches, non_broken_patches

    def create_non_broken_patches(self, num_patches):
        if not self.valid:
            return []
        patches = []
        centers = []
        for a in self.annotations:
            centers.append(a.center)

        i = 0
        x_min,y_min,x_max,y_max = self.get_roi()

        tries = 0

        while i < num_patches and tries < 1000:

            cx = rand.randint(x_min,x_max)
            cy = rand.randint(y_min,y_max)

            if self.valid_non_broken_center(cx, cy, centers):
                centers.append((cx,cy))
                patches += self.create_patches(cx,cy)
                i +=1
            tries += 1

        return patches



    def get_roi(self):
        return self.XmlParser.get_roi()


    def extract_patch(self, img, cx, cy):
        return img[cy - PATCH_SIZE_HALF: cy + PATCH_SIZE_HALF,cx - PATCH_SIZE_HALF: cx + PATCH_SIZE_HALF,:]

    def valid_non_broken_center(self, cx,cy, centers):

        for c in centers:
            if np.linalg.norm(np.array([cx,cy]) - np.array(c)) < PATCH_SIZE_HALF:
                return False
        return True


    def in_roi(self, cx,cy):
        roi = self.roi

        return cx >= roi[0] and cx <= roi[2] and cy >= roi[1] and cy <= roi[3]

    def check_and_fix_roi(self, cx, cy, scale=1 ):
        #TODO make do work
        return cx,cy


    def create_broken_patches(self):

        patches = []
        i = 0
        for a in self.annotations:
            x_off = rand.randint(-OFFSET_RANGE,OFFSET_RANGE)
            y_off = rand.randint(-OFFSET_RANGE,OFFSET_RANGE)
            cx = a.center[0]# + x_off
            cy = a.center[1]# + y_off



            cx, cy = self.check_and_fix_roi(cx,cy)
            if not self.in_roi(cx, cy):
                continue
            i+=1

            patches += self.create_patches(cx,cy)

        return patches, i

    def create_patches(self,cx,cy):

        img = self.img()
        patches = []



        rots = range(0,90,10) if self.settings.rotation else [0]
        scs = np.arange(0.8,1.4,0.2) if self.settings.scale else [1]
        tls = range(-5,5,4) if self.settings.translate else [0]
        gms = np.arange(0.6,1.5,0.4) if self.settings.gamma else [1]

        for angle in rots: # rotation
            for scale in scs: # scale
                for tx in tls: # translate X
                    for ty in tls: # translate Y
                        for gm in gms: # gamma
                            x, y = self.check_and_fix_roi(cx+tx, cy + ty)
                            patch = subimage(img, (x,y), angle, scale, PATCH_SIZE, PATCH_SIZE)
                            patch = adjust_gamma(patch,gm)
                            patches.append(patch)

        return patches

class XmlParser(object):

    def __init__(self, xml_img_path, img_name, frame):
        tree = ET.parse(xml_img_path)
        root = tree.getroot()
        self.xml_root = root
        self.img_name = img_name
        self.frame = frame
        self.valid = True
        self.xml_element = self.get_xml_entry(self.xml_root)

    def get_xml_entry(self, root):
        # get all frames
        self.xml_element = None
        frames = root.findall("frame")
        for child in frames:
            if child.attrib['number'] == self.frame:
                self.xml_element = child


        if self.xml_element == None:
            self.valid = False
            return  None

        if self.xml_element.find("image").attrib["name"] == self.img_name:
            self.xml_element = self.xml_element.find("image").find("graphics")
        return self.xml_element



    def get_annotations(self):
        annotations = []
        if self.xml_element == None:
            self.get_xml_entry(self.xml_root)
        if not self.valid:
            return annotations



        circlesE = self.xml_element.find("circles")

        if circlesE != None:
            circles = circlesE.findall("circle")


            for c in circles:
                annotations.append(Annotation(c))


        return annotations

    def get_roi(self):
        if self.xml_element == None:
            self.get_xml_entry(self.xml_root)
        if not self.valid:
            return 0,0,0,0

        xs = []
        ys = []
        for l in self.xml_element.iter('line'):
            x_start = int(float(l.find('start').find('X').text))
            y_start = int(float(l.find('start').find('Y').text))

            x_end = int(float(l.find('end').find('X').text))
            y_end = int(float(l.find('end').find('Y').text))

            xs += [x_start, x_end]
            ys += [y_start, y_end]

        return np.min(xs) + PATCH_SIZE_HALF, np.min(ys) + PATCH_SIZE_HALF, np.max(xs) - PATCH_SIZE_HALF, np.max(ys) - PATCH_SIZE_HALF




class Annotation(object):

    def __init__(self, xml_circle_element):

        self.size = xml_circle_element.find("size").text
        self.size = int(float(self.size))

        self.radius = xml_circle_element.find('radius').text
        self.radius = int(float(self.radius))

        self.X = xml_circle_element.find("center").find("X").text
        self.X = int(float(self.X))

        self.Y = xml_circle_element.find("center").find("Y").text
        self.Y = int(float(self.Y))
        self.center = (self.X, self.Y)



def adjust_gamma(img, gamma):


    invGamma = 1.0/ gamma

    table = np.array([((i/255.0) ** invGamma) * 255
                      for i in np.arange(0,256)]).astype("uint8")

    return cv2.LUT(img,table)



def get_all_annotations_from_xml(xml_root):
    annos = []
    for cs in xml_root.iter('circles'):
        for c in cs.findall('circle'):
            annos.append(Annotation(c))

    return annos


def get_all_annotations(path):
    annos = []
    for r,ds,fs in os.walk(path):
        for f in fs:
            if f == IMG_META_XML_NAME:
                tree = ET.parse(r + '/' + f)
                root = tree.getroot()
                annos += get_all_annotations_from_xml(root)
    return annos


def subimage_sc(image, center, scale, width, height):


    v_x = (1,0)
    v_y = (0,1)
    s_x = center[0] - v_x[0] * (width / 2) - v_y[0] * (height / 2)
    s_y = center[1] - v_x[1] * (width / 2) - v_y[1] * (height / 2)

    mapping = np.array([[scale * v_x[0], scale * v_y[0], s_x],
                        [scale * v_x[1], scale * v_y[1], s_y]])

    return cv2.warpAffine(image,mapping,(width, height),flags=cv2.WARP_INVERSE_MAP,borderMode=cv2.BORDER_REPLICATE)

def subimage(image, center, theta, scale,  width, height):
   """
   Code from:
   http://stackoverflow.com/questions/11627362/how-to-straighten-a-rotated-rectangle-area-of-an-image-using-opencv-in-python

   http://stackoverflow.com/questions/11627362/how-to-straighten-a-rotated-rectangle-area-of-an-image-using-opencv-in-python
   """
   theta *= 3.14159 / 180 # convert to rad

   v_x = (np.cos(theta), np.sin(theta))
   v_y = (-np.sin(theta), np.cos(theta))
   s_x = center[0] - v_x[0] * (width / 2) - v_y[0] * (height / 2)
   s_y = center[1] - v_x[1] * (width / 2) - v_y[1] * (height / 2)

   mapping = np.array([[scale * v_x[0], scale * v_y[0], s_x],
                        [scale * v_x[1], scale * v_y[1], s_y]])

   return cv2.warpAffine(image,mapping,(width, height),flags=cv2.WARP_INVERSE_MAP,borderMode=cv2.BORDER_REPLICATE)



class Settings(object):

    def __init__(self, rotation,scale, translate, gamma):
        self.rotation = rotation
        self.scale = scale
        self.translate = translate
        self.gamma = gamma

if __name__ == "__main__":

    path = sys.argv[sys.argv.index("path") +1]

    print(path)


    if "count" in sys.argv:
        print(len(get_all_annotations(path)))
