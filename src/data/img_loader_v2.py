import cv2
import xml.etree.ElementTree as ET
import os.path
import json
import utils as UT
import random as rand
import numpy as np

IMG_META_XML_NAME = "Images Metadata Log.xml"
FILE_NAME_SUBSTRING = "all_impurities"

PATCH_SIZE = 64
PATCH_SIZE_HALF = PATCH_SIZE / 2
OFFSET_RANGE = 5
NUM_NON_BROKEN_PATHCES = 10


class XmlImgsLoad(object):

    def __init__(self, path):
        self.path = path

        self.XmlParser = XmlParser(self.path  +'/' + IMG_META_XML_NAME)



    def get_all_img_iter(self, imgs_name):
        """
        given fx the name 'all_impurities' find all images in xml that fits that name
        """


        xml_entries = self.XmlParser.get_xml_entries(imgs_name)

        for xml_entry in xml_entries:
            file_name = xml_entry.find("properties//filename").text

            yield (cv2.imread(self.path + '/' + file_name), xml_entry)



    def get_img_roi(self, imgs_name):
        for (img,xml) in self.get_all_img_iter(imgs_name):

            roi = self.XmlParser.get_roi(xml)


            roi_img = img[roi[1]:roi[3],roi[0]:roi[2],:]


            yield roi_img,xml



class XmlParser(object):

    def __init__(self, xml_img_path):
        tree = ET.parse(xml_img_path)
        root = tree.getroot()
        self.xml_root = root

    def get_xml_entries(self, imgs_name):

        return filter(lambda x : x.attrib['name'] == FILE_NAME_SUBSTRING,  self.xml_root.findall(".//image"))


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

    def get_roi(self, xml_element):

        xs = []
        ys = []
        for l in xml_element.iter('line'):
            x_start = int(float(l.find('start').find('X').text))
            y_start = int(float(l.find('start').find('Y').text))

            x_end = int(float(l.find('end').find('X').text))
            y_end = int(float(l.find('end').find('Y').text))

            xs += [x_start, x_end]
            ys += [y_start, y_end]

        return np.min(xs) , np.min(ys), np.max(xs), np.max(ys)




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


if __name__ == "__main__":

    path = 'E://Speciale//CLAAS//161004_C83-04 Harsewinkel (v02.06.07) Maize//Oktober 4, 2016 - 10 20 56'

    xml_img_loader = XmlImgsLoad(path)

    roi_imgs_xml = xml_img_loader.get_img_roi(FILE_NAME_SUBSTRING)

    for (img,xml) in roi_imgs_xml:
        print xml.find("properties//filename").text
