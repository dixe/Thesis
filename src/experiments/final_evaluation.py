import numpy as np


def calc_score(bin_map, ground_truth):

    bin_white = bin_map[bin_map == 255]

    ground_white = bin_map[bin_map == 255]

    bin_black = bin_map[bin_map == 0]

    ground_black = bin_map[bin_map == 0]

    tp = np.sum(np.logical_and(bin_white, ground_white).flatten())

    tn = np.sum(np.logical_and(bin_black, ground_black).flatten())

    fp = np.sum(np.logical_and(bin_white, ground_black).flatten())

    fn = np.sum(np.logical_and(bin_black, ground_white).flatten())

    # missing number of found annos connected components


    return tp,tn,fp,fn






if __name__ == "__main__":

    test()
