import sys
from keras import optimizers
from keras.models import Sequential, Model
from keras.layers import Input, UpSampling2D, Convolution2D, MaxPooling2D, ZeroPadding2D, RepeatVector, Reshape
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator
from network import Base_network, default_settings
from run_settings import Net_settings
import numpy as np
import Weightstore as ws
import ftlayer as ftl


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

        model = Sequential()

        model.add(ftl.FTLayer(input_shape=(3, self.settings.img_height, self.settings.img_width)))

        conv0 = Convolution2D(3,3,3, activation="sigmoid", border_mode='same')

        conv1 = Convolution2D(30, 3, 3, activation='relu', border_mode='same')



        model.add(conv1)

        model.add(MaxPooling2D((2,2), border_mode="same"))

        model.add(conv0)
        
        model.add(UpSampling2D((2,2)))

        model.add(conv1)

        model.add(conv0)

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




        history = None

        losses = []
        j = 0
        while history == None or self.should_stop(losses):
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


    def should_stop(self, losses):

        return len(losses) < 100

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
