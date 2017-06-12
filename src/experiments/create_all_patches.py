import cv2
from keras.preprocessing.image import load_img, img_to_array

def create_img_patches(img, windows_size, stride):

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



def in_roi(i,j,s,w):
    y = i*s
    x = j*s
    ym = i*s +w
    xm = j*s +w

    return y >= 20 and ym <= 200




if __name__ == "__main__":

    img_name = sys.argv[1]
    img = np.array([img_to_array(load_img(img_name))])

    print "Start create"

    pathces = create_img_patches(img, 64,1)

    print "Finish Create"

    print len(patches)
