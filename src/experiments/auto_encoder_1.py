import sys
from keras import optimizers
from keras.models import Sequential, Model
from keras.layers import Input, UpSampling2D, Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator
from network import Base_network, default_settings
from run_settings import Net_settings
import numpy as np
import Weightstore as ws

class auto_encoder(Base_network):

    def __init__(self,settings):

        Base_network.__init__(self,settings)


    def get_model_test(self):
        model = self.get_model()

        model.compile(loss='binary_crossentropy',
                      optimizer='rmsprop',
                      metrics=['accuracy','recall','precision'])


        model.load_weights(self.settings.save_weights_path)

        return model

    def get_model(self):

        # encoding layer
        input_img = Input(shape=(3, self.settings.img_width, self.settings.img_height))

        conv1 = Convolution2D(32, 3, 3, activation='relu', border_mode='same')
        x = conv1(input_img)

        x = conv1(x)

        decoded = Convolution2D(3,3,3, activation="sigmoid", border_mode='same')(x)

        model = Model(input_img, decoded)



        print  self.has_weights()
        if self.has_weights():
            model.load_weights(self.settings.save_weights_path)
            print "loaded_model"


        return model


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




        for e in range(self.settings.nb_epoch):
            print "Epoche " + str(e)

            imgs = train_generator.next()
            model.fit(imgs, imgs, 32, 1, 1)
            for i in range(self.settings.nb_train_samples/32):
                imgs = train_generator.next()
                model.fit(imgs, imgs, 32, 1, 0)

        return model, None


    def get_model_train(self):
        model = self.get_model()

        model.compile(optimizer='adadelta', loss='binary_crossentropy')

        return model

    def model_name(self):
        return "auto_encoder_1"

    def description(self):
        return "Auto encoder that has same encoder layer settings simple_model"


def train(guid_substring = None):
    settings = ws.get_settings(guid_substring)
    if settings == None:
        settings = default_settings()

    net = auto_encoder(settings)
    net.fine_tune_and_save()

def get_model_test(settings):
    net = auto_encoder(settings)
    return net.get_model_test()

if __name__ == "__main__":
    guid_substring = sys.argv[-1]
    train(guid_substring)
