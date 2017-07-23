import json
import sys
import cv2

DATASET_BASE_PATH = "E:/Speciale/CLAAS/Datasets/arg_data_sets_few_whole/train_val/{0}/validation/{1}"

def get_images_paths(file_name, boundary_thres = -1):
    with open(file_name) as f:
        json_dict = json.load(f)
    dataset = json_dict['dataset']

    imgs_paths = []
    for k in json_dict.keys():
        if k != 'dataset':
            if boundary_thres == -1:
                imgs_paths.append(DATASET_BASE_PATH.format(dataset,k))
            else:
                if abs(json_dict[k] - 0.5) < boundary_thres:
                    imgs_paths.append(DATASET_BASE_PATH.format(dataset,k))


    print len(imgs_paths)
    return imgs_paths



def show_images(imgs_paths):
    for path in imgs_paths:

        img = cv2.imread(path)
        title = "broken" if "broken" in path else "whole"
        print path.split('/')[-1]
        cv2.imshow(title,img)
        cv2.waitKey()




if __name__ == "__main__":
    file_name = sys.argv[1]
    print file_name
    paths = get_images_paths(file_name)
    show_images(paths)
