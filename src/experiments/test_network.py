import sys
import run_settings as rs
import json
import Weightstore as ws
import numpy as np
import cv2
from keras.preprocessing.image import load_img, img_to_array

def visualize_model(net):
    from keras.utils.visualize_util import plot

    plot(model, to_file= sys.argv[2]+'.png')


def evaluate_model_ae(model):
    print "eval"
    from keras.preprocessing.image import ImageDataGenerator

    eval_datagen = ImageDataGenerator(rescale=1./255)

    eval_generator = eval_datagen.flow_from_directory(
        rs.validation_data_dir,
        target_size=(rs.img_height, rs.img_width),
        batch_size=rs.nb_validation_samples,
        class_mode='binary')

    imgs = eval_generator.next()
    x_eval = np.array(imgs[0])

    res = model.evaluate(x_eval, x_eval)
    print res
    return res

def evaluate_model(model):
    print "eval"
    from keras.preprocessing.image import ImageDataGenerator
    eval_datagen = ImageDataGenerator(rescale=1./255)

    eval_generator = eval_datagen.flow_from_directory(
        rs.validation_data_dir,
        target_size=(rs.img_height, rs.img_width),
        batch_size=32,
        class_mode='binary')

    res = model.evaluate_generator(
        eval_generator,
        val_samples=rs.nb_validation_samples)

    print res
    return res


def evaluate_model_and_report(model):
    print "report"
    from keras.preprocessing.image import ImageDataGenerator
    eval_datagen = ImageDataGenerator(rescale=1./255)

    eval_generator = eval_datagen.flow_from_directory(
        rs.validation_data_dir,
        target_size=(rs.img_height, rs.img_width),
        batch_size=32,
        class_mode='binary')

    res = model.predict_generator(
        eval_generator,
        val_samples=rs.nb_validation_samples)


    file_names = eval_generator.filenames

    print len(file_names) == len(res)

    res_dict = {}

    for i in range(len(file_names)):
        if not correct(file_names[i], float(res[i][0])):
            res_dict[file_names[i]] = float(res[i][0])

    print len(res_dict)
    with open('simple_model_1.json','w+') as fp:
        json.dump(res_dict,fp)

    return res

def visualize_weights(net):
    
    model = net.get_model_train()

    weights = model.layers[0].get_weights()
    
    filter_img_size = 6 * len(weights[0][0]) * len(weights[0][0][0])

    filter_img = np.zeros((filter_img_size, filter_img_size, 3))
    weights[0] = weights[0]
    i = 0
    for i in range(6):
        for j in range(6):
            if i*6 + j >= len(weights[0]):
                continue


            norm_weights = (weights[0][i*6+j] - np.min(weights[0][i*6+j])/ (1.0 * np.max(weights[0][i*6+j])-np.min(weights[0][i*6+j])))*255

            print norm_weights

            filter_img[i*3: i*3 + 3,j*3: j*3 + 3,:] = norm_weights
            
    print np.max(np.max(filter_img))
    cv2.imwrite("weights.png",filter_img)



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


def correct(name, pred):
    if name.startswith("broken"):
        return pred < 0.5
    else:
        return pred > 0.5


def get_path(args):
    path = ""
    if 'path' in args:
        i = args.index('path')
        path = args[i+1]

    return path


if __name__ == "__main__":


    callback = evaluate_model
    if len(sys.argv) == 1:
        print "ftc25, ftc18, sm0, sm1, sm2, sm3 ae"
        exit()


    if 'vis' in sys.argv:
        callback = visualize_model

    if 'rep' in sys.argv:
        callback = evaluate_model_and_report


    if 'wei' in sys.argv:
        callback = visualize_weights

    sys.argv = filter(lambda x : x != '',sys.argv )
    guid_substring = sys.argv[-1]

    settings = ws.get_settings(guid_substring)

    path = get_path(sys.argv)

    if 'pred' in sys.argv:
        fun = lambda model : predict_img_path(path,model)
        callback = fun

    if "ftc25" in sys.argv: # fine_tune_conv_25.py
        import fine_tune_conv_25 as ftc

        model = ftc.get_model_test(settings)
        callback(model)
    elif "ftc18" in sys.argv: # fine_tune_conv_18.py
        import fine_tune_conv_18 as ftc

        model = ftc.get_model_test(settings)
        callback(model)
    elif 'sm0' in sys.argv: # simple_model.py
        import simple_model as sm

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
