import fine_tune_conv as ftc
import sys

eval_data_dir = '/home/ltm741/thesis/datasets/data-64-1000-400/validation'
nb_eval_samples = 800

def evaluate_model(model):

    eval_datagen = ImageDataGenerator(rescale=1./255)

    eval_generator = eval_datagen.flow_from_directory(
        eval_data_dir,
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
