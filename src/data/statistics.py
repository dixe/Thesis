import img_loader as IL
import img_loader_v2 as IL2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab


class AnnoStats(object):

    def __init__(self, annos):
        self.annos = annos

        radius = []
        for a in annos:
            radius.append(float(a.radius))

        self.radius = np.array(radius)
        self.diameter = self.radius * 2.0
        print self.radius, self.diameter



    def radius_std(self):
        return np.std(self.radius)

    def average_radius(self):
        return self.average(self.radius)


    def diameter_std(self):
        return np.std(self.diameter)

    def average_diameter(self):
        return self.average(self.diameter)


    def average(self, data):
        return sum(data)/len(data)


    def radius_hist(self):

        bins = range(41)
        sigma = self.radius_std()

        mu = self.average_radius()

        plt.hist(self.radius, bins = bins)
        y = mlab.normpdf( bins, mu, sigma)
        plt.title("Radius histogram")
        plt.xlabel("Radius in pixels")
        plt.ylabel("#number annotations")
        plt.show()

    def diameter_hist(self):

        bins = range(82)
        sigma = self.diameter_std()

        mu = self.average_diameter()

        plt.hist(self.diameter, bins,normed=False)
        y = mlab.normpdf( bins, mu, sigma)
        plt.title("Diameter histogram")
        plt.xlabel("Diameter in pixels")
        plt.ylabel("#number annotations")
        plt.show()


def roi_size():

    path  ='E:/Speciale/CLAAS'

    rois = IL2.get_all_roi(path + '/BG_Sequences_w_ROI_Annotated')

    np_rois =  np.array(rois)

    print "len {0}".format(len(np_rois))
    print "std {0}".format(np.std(np_rois,axis = 0))
    print "avg {0}".format(np.average(np_rois,axis = 0))




if __name__ == "__main__":


    #roi_size()
    #exit()
    path  ='E:/Speciale/CLAAS'
    annos = IL.get_all_annotations(path + '/BG_Sequences_w_ROI_Annotated')

    print "annos {0}".format(len(annos))

    stats = AnnoStats(annos)
    print stats.average_radius(), stats.radius_std()
    stats.radius_hist()
    print stats.average_diameter(), stats.diameter_std()
    stats.diameter_hist()
