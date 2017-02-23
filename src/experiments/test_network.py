import fine_tune_conv as ftc
import sys
from keras.preprocessing.image import ImageDataGenerator
import static_paths as sp

nb_eval_samples = 800
img_width = 64
img_height = 64

def evaluate_model(model):

    eval_datagen = ImageDataGenerator(rescale=1./255)

    eval_generator = eval_datagen.flow_from_directory(
        sp.eval_data_dir,
        target_size=(img_height, img_width),
        batch_size=32,
        class_mode='binary')

    res = model.evaluate_generator(
        eval_generator,
        val_samples=nb_eval_samples)

    print res


if __name__ == "__main__":

    if "ftc" in sys.argv:
        model = ftc.get_model_test()
        evaluate_model(model)
