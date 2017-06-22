import sys
import run_settings as rs
import json
import Weightstore as ws
import numpy as np
import cv2
from keras.preprocessing.image import load_img, img_to_array
import utils as ut
from keras import backend as K

try:
    import PIL.Image as Image
except ImportError:
    import Image

def visualize_model(net):
    from keras.utils.visualize_util import plot
    model = net.get_model_test()

    plot(model, to_file= sys.argv[2]+'.png')


def evaluate_model_ae(net):
    print "eval"

    model = net.get_model_test()

    from keras.preprocessing.image import ImageDataGenerator

    eval_datagen = ImageDataGenerator(rescale=1./255)

    eval_generator = eval_datagen.flow_from_directory(
        net.settnings.validation_data_dir,
        target_size=(rs.img_height, rs.img_width),
        batch_size=rs.nb_validation_samples,
        class_mode='binary')

    imgs = eval_generator.next()
    x_eval = np.array(imgs[0])

    res = model.evaluate(x_eval, x_eval)

    print model.metrics_names
    print res

    return res

def evaluate_model(net):

    print "eval"

    model = net.get_model_test()

    from keras.preprocessing.image import ImageDataGenerator
    eval_datagen = ImageDataGenerator(rescale=1./255)

    eval_generator = eval_datagen.flow_from_directory(
        net.settings.validation_data_dir,
        target_size=(net.settings.img_height, net.settings.img_width),
        batch_size=32,
        class_mode='binary')

    res = model.evaluate_generator(
        eval_generator,
        val_samples=rs.size_dict_val[net.settings.dataset])

    print model.metrics_names
    print res
    return res


def evaluate_model_and_report(net):

    print "report"
    model = net.get_model_test()
    from keras.preprocessing.image import ImageDataGenerator

    eval_datagen = ImageDataGenerator(rescale=1./255)

    eval_generator = eval_datagen.flow_from_directory(
        net.settings.validation_data_dir,
        target_size=(net.settings.img_height, net.settings.img_width),
        batch_size=rs.size_dict_val[net.settings.dataset],
        class_mode='binary',
        shuffle = False) # With shuffel we cannot get the file names

    eval_generator.next()
    file_names = eval_generator.filenames


    #print eval_generator.classes, len(eval_generator.classes)

    target = eval_generator.classes

    res_dict = {}

    res = model.predict_generator(
        eval_generator,
        val_samples=rs.size_dict_val[net.settings.dataset])

    res = res.flatten()
    res_int = np.array(map(round,res.flatten()))


    TP = len(np.where(np.logical_and(np.equal(res_int,0), np.equal( target, 0)))[0])
    TN = len(np.where(np.logical_and(np.equal(res_int,1), np.equal( target, 1)))[0])
    P = len(np.where(res_int == 0)[0])
    N = len(np.where(res_int == 1)[0])

    print TP, TN, P, N

    print "acc = {0}".format((TP + TN) /(P+N*1.0))

    for i in range(len(file_names)):
        name = "broken" if target[i] == 0 else "whole"
        if not correct(name,res[i]):
            res_dict[file_names[i]] = float(res[i])

    print len(res_dict)
    res_dict['dataset'] = net.settings.dataset

    with open('simple_model_{0}.json'.format(net.settings.dataset),'w+') as fp:
        json.dump(res_dict,fp)

    print net.settings.validation_data_dir

    return res

def visualize_weights(net, layer=0):

    model = net.get_model_test()

    weights = model.layers[layer].get_weights()


    img = ut.tile_raster_color(
        weights[0],
        img_shape=(3,3), tile_shape=(10,10),
        tile_spacing=(1,1))


    cv2.imwrite('{0}_layer_{1}.png'.format(net.settings.dataset, layer), img)


def visualize_layer(net, layer):
    model = net.get_model_test()

    activations = get_activations(model, layer)

    print type(activations)




def get_activations(model, layer, X_batch):
    get_activations = K.function([model.layers[0].input, K.learning_phase()], model.layers[layer].output)
    activations = get_activations([X_batch,0])
    return activations


def visualize_all_weights():

    settings = ws.get_settings_model_name("")

    for s in settings:

        setting = ws.get_settings(s[0])

        net = sm.get_net(setting)
        for l in [0,2,4]:
            print l
            visualize_weights(net,l)



def evaluate_all_models():

    model_name = 'simple_model'
    settings = ws.get_settings_model_name(model_name)

    with open("eval_all_{0}.txt".format(model_name), 'a') as f:
        f.write("dataset, loss, acc, recall, precision\n")

    for s in settings:
        setting = ws.get_settings(s[0])

        net = sm.get_net(setting)

        loss, acc, recall, precision = evaluate_model(net)
        with open("eval_all_{0}.txt".format(model_name), 'a') as f:
            f.write("{0}, {1}, {2}, {3}, {4}\n".format(net.settings.dataset,loss, acc, recall, precision))


def predict_img_path(path,net):

    print "Predicting on {0}".format(path)

    img = np.array([img_to_array(load_img(path))])

    size = net.get_input_shape()

    print img.shape

    img_in = img / 255.0

    res = net.predict_img(img_in) * 255

    res_save = np.zeros((1,64,64,3))
    for c in range(3):
        res_save[0,:,:,c] = res[0,c,:,:]

    cv2.imwrite("Predict.png",res_save[0])


def find_error_images(net):

    model = net.get_model_test()

    from keras.preprocessing.image import ImageDataGenerator

    eval_datagen = ImageDataGenerator(rescale=1./255)

    print net.settings.validation_data_dir

    eval_generator = eval_datagen.flow_from_directory(
        net.settings.validation_data_dir,
        target_size=(rs.img_height, rs.img_width),
        batch_size=rs.size_dict_val[net.settings.dataset],
        class_mode='binary',
        shuffle = False) # don't shuffle flow of images


    imgs = eval_generator.next()

    print model.evaluate(imgs[0],imgs[1])

    cor = 0
    print len(imgs[0])
    for i in range(len(imgs[0])):
        name = "broken" if imgs[1][i] == 0 else "whole"
        pred = model.predict(np.array([imgs[0][i]]))
        if correct(name, pred):
            cor += 1
    print cor,"acc = " + str(cor/ (1.0 * len(imgs[0])))







def correct(name, pred):
    if name.startswith("broken"):
        return pred < 0.5
    else:
        return pred >= 0.5


def get_path(args):
    path = ""
    if 'path' in args:
        i = args.index('path')
        path = args[i+1]

    return path


if __name__ == "__main__":


    callback = evaluate_model
    if len(sys.argv) == 1:
        print "ftc25, ftc18, sm, sm1, sm2, sm3 ae"
        exit()


    if 'evala' in sys.argv:
        callback = evaluate_all_models

    if 'vis' in sys.argv:
        callback = visualize_model

    if 'rep' in sys.argv:
        callback = evaluate_model_and_report


    if 'wei' in sys.argv:
        callback = visualize_weights

    if 'weia' in sys.argv:
        callback = visualize_all_weights

    if 'imgs_errors' in sys.argv:
        callback = find_error_images

    if 'vl' in sys.argv:
        layer = 0
        fun = lambda model : visualize_layer(model, layer)
        callback = fun


    sys.argv = filter(lambda x : x != '',sys.argv )
    guid_substring = sys.argv[-1]

    settings = ws.get_settings(guid_substring)

    path = get_path(sys.argv)

    if 'pred' in sys.argv:
        fun = lambda model : predict_img_path(path,model)
        callback = fun

    if 'imgs_errors' in sys.argv:
        callback = find_error_images




    if "ftc25" in sys.argv: # fine_tune_conv_25.py
        import fine_tune_conv_25 as ftc

        model = ftc.get_model_test(settings)
        callback(model)
    elif "ftc18" in sys.argv: # fine_tune_conv_18.py
        import fine_tune_conv_18 as ftc
        model = ftc.get_model_test(settings)
        callback(model)

    elif 'sm' in sys.argv: # simple_model.py
        import simple_model as sm
        if settings is None:
            callback()
            exit()
        net = sm.get_net(settings)
        callback(net)

    elif 'sm1' in sys.argv: # simple_model_1.py
        import simple_model_1 as sm

        model = sm.get_model_test(settings)
        callback(model)
    elif 'sm2' in sys.argv: # simple_model_2.py
        import simple_model_2 as sm

        model = sm.get_model_test(settings)
        callback(model)

    elif 'sm3' in sys.argv: # simple_model_3.py
        import simple_model_3 as sm

        model = sm.get_model_test(settings)
        callback(model)
    elif 'ae0' in sys.argv: # auto_encoder_0.py
        import auto_encoder_0 as ae
        callback = evaluate_model_ae
        model = ae.get_model_test(settings)
        callback(model)

    elif 'ae1' in sys.argv: # auto_encoder_1.py
        import auto_encoder_1 as ae
        net = ae.get_net(settings)
        callback(net)

    elif 'ae2' in sys.argv: # auto_encoder_2.py
        import auto_encoder_2 as ae
        net = ae.get_net(settings)
        callback(net)

    elif 'ae3' in sys.argv: # auto_encoder_3.py
        import auto_encoder_3 as ae
        net = ae.get_net(settings)
        callback(net)
