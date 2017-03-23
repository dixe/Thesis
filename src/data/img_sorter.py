import cv2
import os.path
import utils as UT
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans

IMG_META_XML_NAME = "Images Metadata Log.xml"


def get_all_imgs(path):
    for img,_ in get_all_imgs_files(path):
        yield img

def get_all_imgs_files(path):
    for r,ds,fs in os.walk(path):
        for f in fs:
            if f.endswith('.bmp'):
                yield (cv2.imread(r + '//' + f), f)



def get_avg_colors(imgs):
    colors = []
    for img in imgs:
        total_pixels = img.shape[0]*img.shape[1]
        avg_color = np.sum(np.sum(img,axis=0),axis =0)/ (1.0 * total_pixels)
        colors.append(avg_color)


    return np.array(colors)


def avg_colors_clusters(imgs):
    print "Getting avgs"

    colors = get_avg_colors(imgs)

    print len(colors)

    print "Doing kmeans"
    kmeans = KMeans(n_clusters = 2).fit(colors)

    print "Done"
    return kmeans

def plot_avg_color(imgs):

    colors = get_avg_colors(imgs)

    plot_3d_points(colors)


def plot_3d_points(points):

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')


    ax.scatter(points[:,0],points[:,1], points[:,2])
    ax.set_xlim(0,255)
    ax.set_ylim(0,255)
    ax.set_zlim(0,255)
    plt.show()

def sort_remove_few_kernels(imgs, kmeans, img_path, good_path, bad_path):

    colors = get_avg_colors(imgs)

    try:
        labels = kmeans.predict(colors)
    except Exception as e:
        print "Error at {0}".format(img_path)
        return
    imgs_files = get_all_imgs_files(img_path)

    i = 0
    print "Saving files"


    for img,f in imgs_files:

        if labels[i] == 0:
            cv2.imwrite(good_path +'//' + f, img)

        if labels[i] == 1:
            cv2.imwrite(bad_path + '//' +f, img)

        i+=1




def mkdir(dir_path):

    if not os.path.exists(dir_path):
        os.makedirs(dir_path)



if __name__ == "__main__":

    root = "E://Speciale//CLAAS_roi_imgs//"
    good_root = 'auto_encoder_data'

    bad_root = 'bad_encoder_data'

    kmeans_path = "E://Speciale//CLAAS_roi_imgs//161014_C66-51 USA (v02.06.07) Maize//Oktober 14, 2016 - 14 34 08//"

    kmeans = avg_colors_clusters(get_all_imgs(kmeans_path))

    for r, ds, fs in os.walk(root):
        if IMG_META_XML_NAME in fs:
            img_path = r
            good_path = r.replace("CLAAS_roi_imgs", good_root)
            bad_path = r.replace("CLAAS_roi_imgs", bad_root)


            mkdir(good_path)
            mkdir(bad_path)

            #skip paths that contains files, we done that
            if len(os.listdir(good_path)) > 0:
                continue

            sort_remove_few_kernels(get_all_imgs(img_path), kmeans, img_path, good_path, bad_path)
            """
            print img_path
            print ""
            print good_path
            print ""
            print bad_path


            print "\n\n\n"
            """

    #plot_avg_color(get_all_imgs(path))
    #sorted_imgs = sort_remove_few_kernels(get_all_imgs(path))

    # ERROR FOLDERs = E:\Speciale\CLAAS_roi_imgs\161014_C66-51 USA (v02.06.07) Maize\Oktober 15, 2016 - 16 11 02 ;
