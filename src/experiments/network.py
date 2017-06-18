import os
import sys
import run_settings as rs
import Weightstore as ws
from keras.models import model_from_json
from keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
import keras.backend.tensorflow_backend as KTF
import MyImgGenerator as mig
import callback as mcb

class Base_network(object):
    """
    Base class for nets handles data paths, writing results and weights
    Classification networks should have this as base type
    """


    def __init__(self, net_settings):
        self.settings = net_settings
        self.settings.model_name = self.model_name()


    def save_model_weight(self, model, history):

        results_str = ""

        #Take the last results, which is what the weights set represent
        results_str = "loss: {0:.4f} - acc; {1:.4f} - val_loss: {2:0.4f} - val_acc: {3:.04f}".format(history['loss'][-1], history['acc'][-1], history['val_loss'][-1], history['val_acc'][-1])

        print "Results string: '{0}'".format(results_str)

        model.save_weights(self.settings.save_weights_path)

        ws.store_settings(self.settings, model, history)

        # Add to overview file/table
        print "saved to {0}".format(self.settings.save_weights_path)



    def get_session(self, gpu_fraction=0.3):
        """
        With 8 gb of ram, use ~4 gb
        """

        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction)

        return tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))


    def train_model(self, model):


        # prepare data augmentation configuration
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            #shear_range=0.2,
            #zoom_range=0.2,
            #width_shift_range = 2,
            #height_shift_range= 2,
            horizontal_flip=True,
            vertical_flip=True)

        test_datagen = ImageDataGenerator(rescale=1./255)

        print "Training on dataset: " + self.settings.dataset

        train_generator = train_datagen.flow_from_directory(
            self.settings.train_data_dir,
            target_size=(self.settings.img_height, self.settings.img_width),
            batch_size=32,
            class_mode='binary')

        validation_generator = test_datagen.flow_from_directory(
            self.settings.validation_data_dir,
            target_size=(self.settings.img_height, self.settings.img_width),
            batch_size=32,
            class_mode='binary')


        stop_callback = mcb.EarlyStoppingByLossVal(self.settings.dataset)
        # fine-tune the model
        history = model.fit_generator(
            train_generator,
            nb_epoch=rs.nb_epoch,
            samples_per_epoch = rs.size_dict_train[self.settings.dataset],
            callbacks=[stop_callback],
            validation_data=validation_generator,
            nb_val_samples=self.settings.nb_validation_samples)

        return model, history


    def fine_tune_and_save(self):
        KTF.set_session(self.get_session())

        model = self.get_model_train()

        self.settings.description = self.get_run_description()

        model, history = self.train_model(model)

        self.save_model_weight(model, history.history)


    def get_run_description(self):
        if 'des' in sys.argv:
            return sys.argv[sys.argv.index('des') + 1]

        return ""


    def model_name(self):
        raise NotImplementedError


    def has_weights(self):
        return os.path.isfile(self.settings.save_weights_path)


    def predict_img(self, img):

        model = self.get_model_test()

        return model.predict(img)


    def get_input_shape(self):
        return (self.settings.img_width, self.settings.img_height)


    def set_pretrained_weights(self, model):
        import auto_encoder_2_1 as ae21
        guid_substring = "413c"
        weight_settings = ws.get_settings(guid_substring)

        return model

        ae = ae21.auto_encoder(weight_settings)

        ae_model = ae.get_model()

        print "Settings weights from {0}".format(guid_substring)

        model.layers[0].set_weights(ae_model.layers[0].get_weights())
        model.layers[1].set_weights(ae_model.layers[1].get_weights())

        for l in model.layers:
            print l.trainable

        return model

    def get_model_train(self):
        raise NotImplementedError


    def get_model_test(self):
        raise NotImplementedError


    def load_model(self):
        json_string = ""
        with open ("models/{0}.json".format(str(self.settings.guid)),'r+') as jf:
            json_string = jf.read()

        if not json_string == "":
            model = model_from_json(json_string)
            return

        return None



class Auto_encoder(Base_network):
    """
    Base class autoencoder nets handles target is not a class (num), but an image
    Auto_encoder networks should have this as base type
    """

    def predict_img(self, img):
        model = self.get_model_test()

        res_img = model.predict(img)

        return res_img

    def train_model(self, model):

        nb_batch_size = 32

        train_datagen = ImageDataGenerator(
            rescale=1./255,
            shear_range=0.2,
            zoom_range=0.2,
            rotation_range=360,
            width_shift_range=10,
            height_shift_range=10,
            horizontal_flip=True,
            vertical_flip=True)


        train_generator = mig.MyImgGenerator(train_datagen.flow_from_directory(
            self.settings.train_data_dir,
            batch_size=nb_batch_size,
            target_size=(self.settings.img_height, self.settings.img_width),
            class_mode=None))


        stop_callback = mcb.Stop_callback()
        #fine-tune the model
        history = model.fit_generator(
            train_generator,
            samples_per_epoch=rs.nb_train_samples,
            callbacks=[stop_callback],
            nb_epoch=rs.nb_epoch,
            max_q_size=1)

        return model, history



    def should_continiue(self, losses):
        return len(losses) < 2
        #return len(losses) < 100


def default_settings(dataset = None):

    return rs.default_settings(dataset)
