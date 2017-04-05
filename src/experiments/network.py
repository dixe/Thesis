import run_settings as rs
import Weightstore as ws
from keras import optimizers
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
import keras.backend.tensorflow_backend as KTF
import os, sys
import MyImgGenerator as mig
import h5py

class Base_network(object):
    """
    Base class for nets handles data paths, writing results and weights
    Classification networks should have this as base type
    """


    def __init__(self, net_settings):
        self.settings = net_settings
        self.settings.model_name = self.model_name()


    def save_model_weight(self, model):

        model.save_weights(self.settings.save_weights_path)

        ws.store_settings(self.settings)

        # Add to overview file/table
        print "saved to {0}".format(self.settings.save_weights_path)



    def get_session(self, gpu_fraction=0.99):
        """
        With 8 gb of ram, use ~4 gb
        """

        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction)

        return tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))


    def train_model(self, model):


        # prepare data augmentation configuration
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            shear_range=0.2,
            zoom_range=0.2,
            rotation_range = 360,
            width_shift_range = 10,
            height_shift_range= 10,
            horizontal_flip=True,
            vertical_flip = True)

        test_datagen = ImageDataGenerator(rescale=1./255)

        train_generator = train_datagen.flow_from_directory(
            rs.train_data_dir,
            target_size=(self.settings.img_height, self.settings.img_width),
            batch_size=32,
            class_mode='binary')

        validation_generator = test_datagen.flow_from_directory(
            rs.validation_data_dir,
            target_size=(self.settings.img_height, self.settings.img_width),
            batch_size=32,
            class_mode='binary')

        # fine-tune the model
        history = model.fit_generator(
            train_generator,
            samples_per_epoch=rs.nb_train_samples,
            nb_epoch=rs.nb_epoch,
            validation_data=validation_generator,
            nb_val_samples=self.settings.nb_validation_samples)

        return model, history

    def fine_tune_and_save(self):
        KTF.set_session(self.get_session())

        model = self.get_model_train()

        self.settings.description = self.get_run_description()

        model, history = self.train_model(model)

        self.save_model_weight(model)


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
        return (self.settings.img_width,self.settings.img_height)


    def set_pretrained_weights(self, model):
        import auto_encoder_3 as ae3
        guid_substring = "85df"
        weight_settings = ws.get_settings(guid_substring)

        path = "weights/{0}".format(weight_settings.guid)

        ae = ae3.auto_encoder(weight_settings)

        ae_model = ae.get_model()

        print "Settings weights from {0}".format(guid_substring)

        model.layers[0].set_weights(ae_model.layers[1].get_weights())

        return model


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
            rotation_range = 360,
            width_shift_range = 10,
            height_shift_range= 10,
            horizontal_flip=True,
            vertical_flip = True)


        train_generator = mig.MyImgGenerator(train_datagen.flow_from_directory(
            rs.train_data_dir,
            batch_size = nb_batch_size,
            target_size=(self.settings.img_height, self.settings.img_width),
            class_mode=None))


        #fine-tune the model
        history = model.fit_generator(
            train_generator,
            samples_per_epoch=rs.nb_train_samples,
            nb_epoch=rs.nb_epoch,
            max_q_size=1)

        return model, history


        history = None

        losses = []
        j = 0
        while history == None or self.should_continiue(losses):
            print "Epoche " + str(j)

            imgs = train_generator.next()
            history = model.fit(imgs, imgs, nb_batch_size, 1, 1)
            for i in range(self.settings.nb_train_samples):
                imgs = train_generator.next()
                model.fit(imgs, imgs, nb_batch_size, 1, 0)

            losses.append(history.history['loss'])
            j +=1



        return model, None


    def should_continiue(self, losses):
        return len(losses) < 2
        return len(losses) < 100



def default_settings():
    return rs.Net_settings(rs.img_width,
                        rs.img_height,
                        rs.train_data_dir,
                        rs.validation_data_dir,
                        rs.nb_train_samples,
                        rs.nb_validation_samples,
                        rs.nb_epoch)
