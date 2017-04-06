import sys
from keras import optimizers, regularizers
from keras.models import Sequential, Model
from keras.layers import Input, UpSampling2D, Convolution2D, MaxPooling2D, ZeroPadding2D, RepeatVector, Reshape
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.layers.noise import GaussianNoise
from keras.preprocessing.image import ImageDataGenerator
from network import Base_network, Auto_encoder, default_settings
from run_settings import Net_settings
import numpy as np
import Weightstore as ws


class auto_encoder(Auto_encoder):

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

        model = Sequential()

        model.add(Convolution2D(32, 3, 3, activation='relu', border_mode='same',input_shape=( 3,self.settings.img_height, self.settings.img_width)))

        model.add(Flatten())

        #Intermedite layer

        model.add(Dense(1024, activation='relu', activity_regularizer=regularizers.l1(10e-5)))


        model.add(Dense(self.settings.img_width * self.settings.img_height * 3,activation='sigmoid'))

        model.add(Reshape((3, self.settings.img_height, self.settings.img_width)))

        for l in model.layers:
            print l.input_shape, l.output_shape


        print self.has_weights()

        if self.has_weights():
            model.load_weights(self.settings.save_weights_path)
            print "loaded_model"


        return model


    def get_model_train(self):
        model = self.get_model()

        sgd = optimizers.SGD(lr=0.1, decay=1e-6, momentum=0.9)

        model.compile(optimizer='adadelta', loss='binary_crossentropy', metrics=['accuracy'])

        return model

    def model_name(self):
        return "auto_encoder_4"


    def description(self):
        return "Auto encoder that has same encoder layer settings simple_model, now with sparsety"


def train(guid_substring = None):
    settings = ws.get_settings(guid_substring)
    if settings == None:
        settings = default_settings()

    net = auto_encoder(settings)
    net.fine_tune_and_save()

def get_net(settings):
    return auto_encoder(settings)

if __name__ == "__main__":
    sys.argv = filter(lambda x : x != '',sys.argv )
    guid_substring = sys.argv[-1]
    train(guid_substring)
