import sys
import run_settings as rs
import json


def visualize_model(model):
    from keras.utils.visualize_util import plot

    plot(model, to_file= sys.argv[2]+'.png')


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

def correct(name, pred):
    if name.startswith("broken"):
        return pred < 0.5
    else:
        return pred > 0.5


if __name__ == "__main__":




    callback = evaluate_model
    if len(sys.argv) == 1:
        print "ftc25, ftc18, fsm0, fsm1, fsm2, fsm3"
        exit()

    if 'vis' in sys.argv:
        callback = visualize_model

    if 'rep' in sys.argv:
        callback = evaluate_model_and_report


    if "ftc25" in sys.argv: # fine_tune_conv_25.py
        import fine_tune_conv_25 as ftc

        model = ftc.get_model_test()
        callback(model)
    elif "ftc18" in sys.argv: # fine_tune_conv_18.py
        import fine_tune_conv_18 as ftc

        model = ftc.get_model_test()
        callback(model)
    elif 'fsm0' in sys.argv: # simple_model.py
        import simple_model as fsm

        model = fsm.get_model_test()
        callback(model)

    elif 'fsm1' in sys.argv: # simple_model_1.py
        import simple_model_1 as fsm

        model = fsm.get_model_test()
        callback(model)
    elif 'fsm2' in sys.argv: # simple_model_2.py
        import simple_model_2 as fsm

        model = fsm.get_model_test()
        callback(model)

    elif 'fsm3' in sys.argv: # simple_model_3.py
        import simple_model_3 as fsm

        model = fsm.get_model_test()
        callback(model)
