""" This file contains different utility functions that are not connected
in anyway to the networks presented in the tutorials, but rather help in
processing the outputs into a more understandable way.

For example ``tile_raster_images`` helps in generating a easy to grasp
image from a set of samples or weights.


CODE ORIGINAL FROM http://deeplearning.net/tutorial/utilities.html#how-to-plot
"""

import numpy as np
import cv2

def scale_to_unit_interval(ndar, eps=1e-8):
    """ Scales all values in the ndarray ndar to be between 0 and 1 """
    ndar = ndar.copy()
    ndar -= ndar.min()
    ndar *= 1.0 / (ndar.max() + eps)
    return ndar


def tile_raster_images(X, img_shape, tile_shape, tile_spacing=(0, 0),
                       scale_rows_to_unit_interval=True,
                       output_pixel_vals=True):
    """
    Transform an array with one flattened image per row, into an array in
    which images are reshaped and layed out like tiles on a floor.

    This function is useful for visualizing datasets whose rows are images,
    and also columns of matrices for transforming those rows
    (such as the first layer of a neural net).

    :type X: a 2-D ndarray or a tuple of 4 channels, elements of which can
    be 2-D ndarrays or None;
    :param X: a 2-D array in which every row is a flattened image.

    :type img_shape: tuple; (height, width)
    :param img_shape: the original shape of each image

    :type tile_shape: tuple; (rows, cols)
    :param tile_shape: the number of images to tile (rows, cols)

    :param output_pixel_vals: if output should be pixel values (i.e. int8
    values) or floats

    :param scale_rows_to_unit_interval: if the values need to be scaled before
    being plotted to [0,1] or not


    :returns: array suitable for viewing as an image.
    (See:`Image.fromarray`.)
    :rtype: a 2-d array with same dtype as X.

    """

    assert len(img_shape) == 2
    assert len(tile_shape) == 2
    assert len(tile_spacing) == 2

    # The expression below can be re-written in a more C style as
    # follows :
    #
    # out_shape    = [0,0]
    # out_shape[0] = (img_shape[0]+tile_spacing[0])*tile_shape[0] -
    #                tile_spacing[0]
    # out_shape[1] = (img_shape[1]+tile_spacing[1])*tile_shape[1] -
    #                tile_spacing[1]
    out_shape = [
        (ishp + tsp) * tshp - tsp
        for ishp, tshp, tsp in zip(img_shape, tile_shape, tile_spacing)
    ]

    if isinstance(X, tuple):
        assert len(X) == 4
        # Create an output numpy ndarray to store the image
        if output_pixel_vals:
            out_array = np.zeros((out_shape[0], out_shape[1], 4),
                                    dtype='uint8')
        else:
            out_array = np.zeros((out_shape[0], out_shape[1], 4),
                                    dtype=X.dtype)

        #colors default to 0, alpha defaults to 1 (opaque)
        if output_pixel_vals:
            channel_defaults = [0, 0, 0, 255]
        else:
            channel_defaults = [0., 0., 0., 1.]

        for i in range(4):
            if X[i] is None:
                # if channel is None, fill it with zeros of the correct
                # dtype
                dt = out_array.dtype
                if output_pixel_vals:
                    dt = 'uint8'
                out_array[:, :, i] = np.zeros(
                    out_shape,
                    dtype=dt
                ) + channel_defaults[i]
            else:
                # use a recurrent call to compute the channel and store it
                # in the output
                out_array[:, :, i] = tile_raster_images(
                    X[i], img_shape, tile_shape, tile_spacing,
                    scale_rows_to_unit_interval, output_pixel_vals)
        return out_array

    else:
        # if we are dealing with only one channel
        H, W = img_shape
        Hs, Ws = tile_spacing

        # generate a matrix to store the output
        dt = X.dtype
        if output_pixel_vals:
            dt = 'uint8'
        out_array = np.zeros(out_shape, dtype=dt)

        for tile_row in range(tile_shape[0]):
            for tile_col in range(tile_shape[1]):
                if tile_row * tile_shape[1] + tile_col < X.shape[0]:
                    this_x = X[tile_row * tile_shape[1] + tile_col]
                    if scale_rows_to_unit_interval:
                        # if we should scale values to be between 0 and 1
                        # do this by calling the `scale_to_unit_interval`
                        # function
                        this_img = scale_to_unit_interval(
                            this_x.reshape(img_shape))
                    else:
                        this_img = this_x.reshape(img_shape)
                    # add the slice to the corresponding position in the
                    # output array
                    c = 1
                    if output_pixel_vals:
                        c = 255
                    out_array[
                        tile_row * (H + Hs): tile_row * (H + Hs) + H,
                        tile_col * (W + Ws): tile_col * (W + Ws) + W
                    ] = this_img * c
        return out_array



def tile_raster_color(weights, img_shape, tile_shape, tile_spacing=(1, 1)):
    out_shape    = [0,0]
    out_shape[0] = (img_shape[0]+tile_spacing[0])*tile_shape[0] - tile_spacing[0]
    out_shape[1] = (img_shape[1]+tile_spacing[1])*tile_shape[1] - tile_spacing[1]

    imgs = len(weights[0])

    cols = int(np.sqrt(imgs))

    rows = imgs/cols

    out_img = np.zeros((rows * img_shape[0] * tile_shape[0], cols * img_shape[1] * tile_shape[1],3))

    out_img = np.zeros((cols * img_shape[0] * tile_shape[0] + cols * tile_spacing[0] + tile_spacing[0] , rows * img_shape[1]  * tile_shape[1]+ rows * tile_spacing[1] + tile_spacing[1] ,3))

    for i in range(imgs):

        filteri = np.zeros((img_shape[0] * tile_shape[0], img_shape[1] * tile_shape[1] ,3))

        for h in range(len(weights[i])):
            for j in range(len(weights[i][0])):
                filteri[h * tile_shape[0]: h  * tile_shape[0] + tile_shape[0],
                        j * tile_shape[1]: j  * tile_shape[1] +  tile_shape[1]] = weights[i,h,j,:]


        col = int(i / rows)

        offseti = i*img_shape[1] *tile_shape[1] + tile_spacing[1] * i + tile_shape[1] * img_shape[1]

        offsetcol = col*img_shape[0] *tile_shape[0] + tile_spacing[0] * col + tile_shape[0] * img_shape[0]

        #print filteri.shape

        out_img[tile_spacing[1] + col *img_shape[0]*tile_shape[0] + col * tile_spacing[1]: offsetcol + tile_spacing[0], tile_spacing[1]+i  *img_shape[1]*tile_shape[1] + i * tile_spacing[1] : offseti + tile_spacing[1]] = filteri

    print out_img.shape

    return out_img




if __name__ == "__main__":

    X = np.array([
        [[[255,255,255],[200,20,220], [128,128,128]],
         [[255,255,255],[220,220,220], [128,128,128]],
         [[255,255,255],[220,220,220], [128,128,128]]],

        [[[0,0,0],[0,220,0], [128,128,128]],
         [[255,0,0],[022,0,0], [128,128,128]],
         [ [255,255,0],[022,0,0], [128,128,128]]],

        [[[255,255,255],[110,0,0], [128,128,128]],
         [[255,0,0],[0,0,220], [128,128,128]],
         [[255,255,0],[220,0,0], [128,128,128]]]])


    out_img = tile_raster_color(X,(3,3), (10,10),(1,1))
    cv2.imwrite("out.png", out_img)
