import sys
import os
import numpy as np
import run_settings as rs
import Weightstore as ws
import cv2
from timeit import default_timer as timer
from keras.preprocessing.image import load_img, img_to_array
import final_evaluation as FA

total_pred_time = 0
total_gen_time = 0

def predict_img(model, img, img_name, root, window_size = 64, stride = 4, bin_map = False):


    global total_pred_time
    global total_gen_time

    strides_x = len(img[0][0]) / stride
    strides_y = len(img[0][0][0])/ stride

    preds = []

    #shape = patch.shape
    c = 0
    res_img = None
    if bin_map:
        res_img = np.zeros((len(img[0][0]),len(img[0][0][0])), dtype = np.uint8)
    else:
        # draw on top of regular img
        res_img = np.zeros((len(img[0][0]),len(img[0][0][0]),4))

        res_img[:,:,2] = img[0,0,:,:]
        res_img[:,:,1] = img[0,1,:,:]
        res_img[:,:,0] = img[0,2,:,:]
        res_img[:,:,3] = 255



    c = 0

    start = timer()
    patches, cords = create_img_patches(img, window_size, stride)
    end = timer()

    total_gen_time += end-start

    if not bin_map:
        print "time to create data", total_gen_time, patches.shape
        print "Starting pred"

    start = timer()
    preds = model.predict(patches)
    end = timer()
    total_pred_time += end-start

    if not bin_map:
        print "Finished preds in", total_pred_time


    if len(cords) != len(preds):
        print "Coordinates and prediction shape does not match"
        exit()

    for i in range(len(cords)):

        if preds[i] <= 0.5:
            x,y = cords[i]
            if bin_map:
                res_img[x,y] = 255
            else:
                res_img[x,y,:] = np.array([0,0,225,128])


            #TODO FIX to also work for 32 patches
            #minx = x - 32
            #maxx = x + 31
            #miny = y - 32
            #maxy = y + 31
            #res_img[minx:maxx,miny,:] = np.array([0,255,0,128])
            #res_img[minx:maxx,maxy,:] = np.array([0,255,0,128])

            #res_img[minx,miny:maxy,:] = np.array([0,255,0,128])
            #res_img[maxx,miny:maxy,:] = np.array([0,255,0,128])


            #store_patch(patch, "{0}/patch_{1}_{2}.png".format(root,img_name,c))

    if bin_map:
        # don't write image when generating bin_maps
        return res_img, total_gen_time, total_pred_time

    cv2.imwrite("{0}/{1}_output.{2}".format(root, img_name.split('.')[0],"png"), res_img)
    print sum(preds)




def create_img_patches(img, window_size, stride):

    strides_x = len(img[0][0]) / stride
    strides_y = len(img[0][0][0])/ stride


    patches = []
    cords = []
    c = 0
    for i in range(strides_x):
        for j in range(strides_y):
            if not in_roi(i,j,stride, window_size):
                continue

            patch = img[:,:,i*stride:i*stride + window_size, j*stride:j*stride+window_size]

            if patch.shape == (1,3,window_size,window_size):
                patches.append(patch)
                cords.append((i*stride + window_size/2, j*stride+window_size/2))

    patches = np.squeeze(np.array(patches))
    cords = np.array(cords)
    return patches, cords



def in_roi(i,j,s,w):
    y = i*s
    x = j*s
    ym = i*s +w
    xm = j*s +w

    return y >= 20 and ym <= 200



def store_patch(patch, name):

    res_patch = np.zeros((64,64,3))

    res_patch[:,:,2] = patch[0,0,:,:]
    res_patch[:,:,1] = patch[0,1,:,:]
    res_patch[:,:,0] = patch[0,2,:,:]

    cv2.imwrite(name,res_patch)





def get_settings_from_sysarg():
    sys.argv = filter(lambda x : x != '',sys.argv )
    guid_substring = sys.argv[-1]
    return ws.get_settings(guid_substring)


def get_arg_from_sysargv(arg_name):
    if arg_name not in sys.argv:
        return Non

    index = sys.argv.index(arg_name)
    return sys.argv[index + 1]



def test_simple(net):

    model = net.get_model_test()

    print rs.full_imgs_path
    for r,fs,fs in os.walk(rs.full_imgs_path):
        for f in fs:
            if f.endswith('impurities.bmps'):

                path = r + "/" + f

                print path

                img = np.array([img_to_array(load_img(path))])

                predict_img(model, img, net.settings.model_name + "_" + f, r)


def run_all_settings():
    global total_pred_time
    global total_gen_time

    img_name = get_arg_from_sysargv('img')
    img = np.array([img_to_array(load_img(img_name))])

    print img_name
    settings = ws.get_settings_model_name("")

    for s in settings:
        print s[0]

        setting = ws.get_settings(s[0])


        net = sm.get_net(setting)
        model = net.get_model_test()

        dataset_n = net.settings.validation_data_dir.split('/')[-2]

        predict_img(model, img, dataset_n, ".", net.settings.img_width)



    avg_gen_time = total_gen_time / (1.0*len(settings))
    print "avg gen time", avg_gen_time

    avg_pred_time = total_pred_time / (1.0*len(settings))
    print "avg pred time", avg_pred_time





def load_model(setting):

 
    if 'simple_model' == setting.model_name:
        import simple_model as sm
    elif 'simple_model_7_5_5' == setting.model_name:
        import simple_model_7_5_5 as sm
    elif 'simple_model_7_fully_drop' == setting.model_name:
        import simple_model_fully_drop as sm
    elif 'simple_model_7_2_layer' == setting.model_name:
        import simple_model_7_2_layer as sm
    elif 'simple_model_7_nomax' == setting.model_name:
        import simple_model_7_nomax as sm


def compare_setting(net, img_path):

    # take a settings and run through all images and compare to ground thruth

    model = net.get_model_test()
    for r,ds,fs in os.walk(img_path):
        scores = []
        gen_times = []
        pred_times = []
        print r
        for f in fs:
            if f.endswith("all_impurities.bmp"):


                img = np.array([img_to_array(load_img(r + '/' + f))])

                #TODO Generate ground truth bin_maps - needs annotation first

                bin_map, total_gen_time, total_pred_time = predict_img(model, img, None, None, window_size = net.settings.img_width, stride = 1, bin_map = True)

                # calc score based on image ground truth
                ground_truth = cv2.imread(r+ "/" + f.replace('.bmp', "_ground_truth.bmp"),0)

                score = FA.calc_score(bin_map, ground_truth)
                scores.append(score)
                gen_times.append(total_gen_time)
                pred_times.append(total_pred_time)

        if len(scores) > 0:
            with open(r + "/results.csv", 'w') as rf:
                rf.write("tp,tn,fp,fn, gen_time, pred_time\n")

                for i in range(len(scores)):
                    score = scores[i]
                    rf.write("{0}, {1}, {2}, {3}, {4}, {5}\n".format(score[0], score[1], score[2], score[3], gen_times[i], pred_times[i]))





def run_multiple_settings(settings, img_path):
    # get settigns as a list of guid strings

    for s in settings:
        setting = ws.get_settings(s)

 
        if 'simple_model' == setting.model_name:
            import simple_model as sm
        elif 'simple_model_7_5_5' == setting.model_name:
            import simple_model_7_5_5 as sm
        elif 'simple_model_7_fully_drop' == setting.model_name:
            import simple_model_fully_drop as sm
        elif 'simple_model_7_2_layer' == setting.model_name:
            import simple_model_7_2_layer as sm
        elif 'simple_model_7_nomax' == setting.model_name:
            import simple_model_7_nomax as sm

        net = sm.get_net(setting)
        compare_setting(net, img_path)





if __name__ == "__main__":


    if 'ft' in sys.argv:
        settings = ['822b4']
        path = "/home/ltm741/thesis/datasets/final_test_sets/mini_test/"
        run_multiple_settings(settings, path)
        exit()


    if 'sm' in sys.argv:
        import simple_model as sm
    elif 'sm755' in sys.argv:
        import simple_model_7_5_5 as sm
    elif 'sm7fd' in sys.argv:
        import simple_model_7_fully_drop as sm
    elif 'sm72' in sys.argv:
        import simple_model_7_2_layer as sm
    elif 'sm7nm' in sys.argv:
        import simple_model_7_nomax as sm


    if 'all' in sys.argv and 'img' in sys.argv:
        run_all_settings()
        exit()

    settings = get_settings_from_sysarg()

    net = sm.get_net(settings)

    if 'img' in sys.argv:
        print "running on single img"

        img_name = get_arg_from_sysargv('img')
        img = np.array([img_to_array(load_img(img_name))])
        model = net.get_model_test()

        dataset_n = net.settings.validation_data_dir.split('/')[-2]

        predict_img(model, img, net.settings.model_name + "_" + dataset_n, ".", net.settings.img_width)
        exit()


    test_simple(net)
