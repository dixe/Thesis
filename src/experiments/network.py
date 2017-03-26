import run_settings as rs
import Weightstore as ws
from keras import optimizers
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
import keras.backend.tensorflow_backend as KTF
import os

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



    def get_session(self, gpu_fraction=0.4):
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
            horizontal_flip=True)

        test_datagen = ImageDataGenerator(rescale=1./255)

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

        # fine-tune the model
        history = model.fit_generator(
            train_generator,
            samples_per_epoch=self.settings.nb_train_samples,
            nb_epoch=self.settings.nb_epoch,
            validation_data=validation_generator,
            nb_val_samples=self.settings.nb_validation_samples)

        return model, history

    def fine_tune_and_save(self):
        KTF.set_session(self.get_session())
        model = self.get_model_train()
        model, history = self.train_model(model)
        self.save_model_weight(model)


    def model_name(self):
        raise NotImplementedError


    def has_weights(self):
        return os.path.isfile(self.settings.save_weights_path)


    def predict_img(self, img):

        model = self.get_model_test()




        return model.predict(img)


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
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True)

        train_generator = train_datagen.flow_from_directory(
            self.settings.train_data_dir,
            target_size=(self.settings.img_height, self.settings.img_width),
            class_mode=None)

        history = None

        losses = []
        j = 0
        while history == None or self.should_continiue(losses):
            print "Epoche " + str(j)

            imgs = train_generator.next()
            history = model.fit(imgs, imgs, 32, 1, 1)
            for i in range(self.settings.nb_train_samples/32):
                imgs = train_generator.next()
                model.fit(imgs, imgs, 32, 1, 0)

            losses.append(history.history['loss'])
            j +=1



        for i in range(len(model.layers[-2].get_weights())):
            print (model.layers[-2].get_weights()[i] == model.layers[-6].get_weights()[i]).all()
        return model, None


    def should_continiue(self, losses):

        return len(losses) < 2


def default_settings():
    return rs.Net_settings(rs.img_width,
                        rs.img_height,
                        rs.train_data_dir,
                        rs.validation_data_dir,
                        rs.nb_train_samples,
                        rs.nb_validation_samples,
                        rs.nb_epoch)
