import sys
import run_settings as rs



def visualize_model(model):
    from keras.utils.visualize_util import plot

    plot(model, to_file= sys.argv[2]+'.png')
    exit()

def evaluate_model(model, vis):
    if vis:
        visualize_model(model)

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


if __name__ == "__main__":

    if len(sys.argv) == 1:
        print "ftc25, ftc18, fsm"
        exit()

    vis = 'vis' in sys.argv


    if "ftc25" in sys.argv: # fine_tune_conv_25.py
        import fine_tune_conv_25 as ftc

        model = ftc.get_model_test()
        evaluate_model(model, vis)
    elif "ftc18" in sys.argv: # fine_tune_conv_18.py
        import fine_tune_conv_18 as ftc

        model = ftc.get_model_test()
        evaluate_model(model, vis)
    elif 'fsm' in sys.argv: # ex-64-1000-400.py
        import simple_model as fsm

        model = fsm.get_model_test()
        evaluate_model(model, vis)
