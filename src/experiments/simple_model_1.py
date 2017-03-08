'''This script goes along the blog post
"Building powerful image classification models using very little data"
from blog.keras.io.

It uses data that can be downloaded at:
https://www.kaggle.com/c/dogs-vs-cats/data

In our setup, we:
5B5B- created a data/ folder
- created train/ and validation/ subfolders inside data/
- created cats/ and dogs/ subfolders inside train/ and validation/
- put the cat pictures index 0-999 in data/train/cats
- put the cat pictures index 1000-1400 in data/validation/cats
- put the dogs pictures index 12500-13499 in data/train/dogs
- put the dog pictures index 13500-13900 in data/validation/dogs

So that we have 1000 training examples for each class, and 400 validation examples for each class.


'''

from keras import optimizers
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.layers import Activation, Dropout, Flatten, Dense
from network import Net_settings, Base_network, default_settings


class simple_model(Base_network):

    def __init__(self,settings):

        Base_network.__init__(self,settings)


    def get_model_test(self):
        model = self.get_model()

        model.compile(loss='binary_crossentropy',
                      optimizer='rmsprop',
                      metrics=['accuracy','recall','precision'])


        model.load_weights(self.get_save_weights_path())

        return model

    def get_model(self):

        model = Sequential()
        model.add(Convolution2D(16, 3, 3, input_shape=(3, self.settings.img_width, self.settings.img_height)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Convolution2D(16, 3, 3))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Convolution2D(32, 3, 3))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Flatten())
        model.add(Dense(64))
        model.add(Activation('relu'))
        model.add(Dropout(0.5))
        model.add(Dense(1))
        model.add(Activation('sigmoid'))

        #model.load_weights(self.get_save_weights_path())

        return model

    def get_model_train(self):
        model = self.get_model()
        # compile the model with a SGD/momentum optimizer
        # and a very slow learning rate.
        model.compile(loss='binary_crossentropy',
                      optimizer='rmsprop',
                      metrics=['hinge'])
        return model

    def get_save_weights_path(self):
        return self.settings.save_weights_path.format(self.model_name())

    def model_name(self):
        return "simple_model_1"

def train():
    settings = default_settings()
    net = simple_model(settings)
    net.fine_tune_and_save()

def get_model_test(guid_substring):
    settings = ws.load_settings(guid_substring)
    num_settings = len(settings)
    if len(num_settings) != 1:
        if num_settings == 0:
            print "No settings found: {0}".format(guid_substring)
            exit()

        print "Multiple settings found"
        for s in settings:
            print s.guid
        exit()

    net = simple_model(settings)
    return net.get_model_test()

if __name__ == "__main__":
    train()
