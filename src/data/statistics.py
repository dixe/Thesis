import img_loader as IL
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

        plt.hist(self.radius, bins = bins, normed=True)
        y = mlab.normpdf( bins, mu, sigma)
        l = plt.plot(bins,y,'r--')
        plt.title("Radius histogram")
        plt.show()

    def diameter_hist(self):

        bins = range(82)
        sigma = self.diameter_std()

        mu = self.average_diameter()

        plt.hist(self.diameter, bins,normed=True)
        y = mlab.normpdf( bins, mu, sigma)
        l = plt.plot(bins,y,'r--')
        plt.title("Diameter histogram")
        plt.show()


if __name__ == "__main__":
    path  ='E:/Speciale/CLAAS'
    annos = IL.get_all_annotations(path + '/BG_Sequences_w_ROI_Annotated')

    stats = AnnoStats(annos)
    print stats.average_radius(), stats.radius_std()
    stats.radius_hist()
    print stats.average_diameter(), stats.diameter_std()
    stats.diameter_hist()
