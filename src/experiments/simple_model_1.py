
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


        model.load_weights(self.setings.get_save_weights_path)

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

        print  self.has_weights()
        if self.has_weights():
            model.load_weights(self.settings.save_weights_path)
            print "loaded_model"


        return model

    def get_model_train(self):
        model = self.get_model()
        # compile the model with a SGD/momentum optimizer
        # and a very slow learning rate.
        model.compile(loss='binary_crossentropy',
                      optimizer='rmsprop',
                      metrics=['hinge'])
        return model

    def model_name(self):
        return "simple_model_1"


def train(guid_substring = None):
    settings = ws.get_settings(guid_substring)
    if settings == None:
        settings = default_settings()

    net = simple_model(settings)
    net.fine_tune_and_save()

def get_model_test(guid_substring):
    settings = ws.get_settings(guid_substring)
    if settings == None:
        exit()
    net = simple_model(settings)
    return net.get_model_test()


if __name__ == "__main__":
    guid_substring = sys.argv[-1]
    train(guid_substring)
