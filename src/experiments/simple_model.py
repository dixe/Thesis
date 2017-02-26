'''This script goes along the blog post
"Building powerful image classification models using very little data"
from blog.keras.io.

It uses data that can be downloaded at:
https://www.kaggle.com/c/dogs-vs-cats/data

In our setup, we:
- created a data/ folder
- created train/ and validation/ subfolders inside data/
- created cats/ and dogs/ subfolders inside train/ and validation/
- put the cat pictures index 0-999 in data/train/cats
- put the cat pictures index 1000-1400 in data/validation/cats
- put the dogs pictures index 12500-13499 in data/train/dogs
- put the dog pictures index 13500-13900 in data/validation/dogs

So that we have 1000 training examples for each class, and 400 validation examples for each class.


'''

from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from network import Net_settings, Base_network, default_settings

# dimensions of our images.
img_width, img_height = 64, 64

train_data_dir = '/home/ltm741/thesis/datasets/data-64-1000-400/train'
validation_data_dir = '/home/ltm741/thesis/datasets/data-64-1000-400/validation'
nb_train_samples = 2000
nb_validation_samples = 800
nb_epoch = 1

class simple_model(Base_network):
    def get_model_test():
        model = get_model()

        model.compile(loss='binary_crossentropy',
                      optimizer='rmsprop',
                      metrics=['accuracy','recall','precision'])



        model.load_weights(self.settings.load_weights)

        return model

    def get_model():

        model = Sequential()
        model.add(Convolution2D(32, 3, 3, input_shape=(3, img_width, img_height)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Convolution2D(32, 3, 3))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Convolution2D(64, 3, 3))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Flatten())
        model.add(Dense(64))
        model.add(Activation('relu'))
        model.add(Dropout(0.5))
        model.add(Dense(1))
        model.add(Activation('sigmoid'))

        model.load_weights(self.settings.load_weights)

        return model
