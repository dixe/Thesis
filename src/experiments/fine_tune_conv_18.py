'''This script goes along the blog post
"Building powerful image classification models using very little data"
from blog.keras.io.

It uses data that can be downloaded at:
https://www.kaggle.com/c/dogs-vs-cats/data

In our setup, we:
- created a data/ folder
- created train/ and validation/ subfolders inside data/
- created cats/ and dogs/ subfolders inside train/ and validation/
'''

import os
import h5py
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.layers import Activation, Dropout, Flatten, Dense
from network import Net_settings, Base_network, default_settings

class fine_tune_conv_18(Base_network):

    def __init__(self,settings):

        Base_network.__init__(self,settings)


    def get_model_test(self):
        model = get_model()

        model.compile(loss='binary_crossentropy',
                      optimizer='rmsprop',
                      metrics=['accuracy','recall','precision'])


        # load the weights save after training
        model.load_weights(self.settings.save_weights_path)

        return model


    def get_model(self):
        # build the VGG16 network
        model = Sequential()
        model.add(ZeroPadding2D((1, 1), input_shape=(3, self.settings.img_width, self.settings.img_height)))

        model.add(Convolution2D(64, 3, 3, activation='relu', name='conv1_1'))
        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(64, 3, 3, activation='relu', name='conv1_2'))
        model.add(MaxPooling2D((2, 2), strides=(2, 2)))

        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(128, 3, 3, activation='relu', name='conv2_1'))
        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(128, 3, 3, activation='relu', name='conv2_2'))
        model.add(MaxPooling2D((2, 2), strides=(2, 2)))

        #10
        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(256, 3, 3, activation='relu', name='conv3_1'))
        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(256, 3, 3, activation='relu', name='conv3_2'))
        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(256, 3, 3, activation='relu', name='conv3_3'))
        model.add(MaxPooling2D((2, 2), strides=(2, 2)))

        #17
        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(512, 3, 3, activation='relu', name='conv4_1'))
        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(512, 3, 3, activation='relu', name='conv4_2'))
        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(512, 3, 3, activation='relu', name='conv4_3'))
        model.add(MaxPooling2D((2, 2), strides=(2, 2)))

        #24
        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(512, 3, 3, activation='relu', name='conv5_1'))
        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(512, 3, 3, activation='relu', name='conv5_2'))
        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(512, 3, 3, activation='relu', name='conv5_3'))
        model.add(MaxPooling2D((2, 2), strides=(2, 2)))
        #31


        # build a classifier model to put on top of the convolutional model
        top_model = Sequential()
        top_model.add(Flatten(input_shape=model.output_shape[1:]))
        top_model.add(Dense(256, activation='relu'))
        top_model.add(Dropout(0.5))
        top_model.add(Dense(1, activation='sigmoid'))

        # add the model on top of the convolutional base
        model.add(top_model)
        # load weights for finetuned model 25
        assert os.path.exists(self.settings.load_weights_path), 'Model weights not found (see "weights_path" variable in script).'

        model.load_weights(self.settings.load_weights_path)
        print 'Model loaded.'


        # set the first 18 layers (up to the second last conv block)
        # to non-trainable (weights will not be updated)
        for layer in model.layers[:18]:
            layer.trainable = False

        return model


    def train_model(self, model):

        # compile the model with a SGD/momentum optimizer
        # and a very slow learning rate.
        model.compile(loss='binary_crossentropy',
                      optimizer=optimizers.SGD(lr=1e-4, momentum=0.9),
                      metrics=['accuracy',])

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
        model.fit_generator(
            train_generator,
            samples_per_epoch=self.settings.nb_train_samples,
            nb_epoch=self.settings.nb_epoch,
            validation_data=validation_generator,
            nb_val_samples=self.settings.nb_validation_samples)

        return model

    def fine_tune_and_save(self):
        model = self.get_model()
        model = self.train_model(model)
        self.save_model_weight(model)


if __name__ == "__main__":
    settings = default_settings()
    net = fine_tune_conv_18(settings)
    net.fine_tune_and_save()
