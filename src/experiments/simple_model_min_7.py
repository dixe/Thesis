import sys
from keras import optimizers
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.layers import Activation, Dropout, Flatten, Dense
from network import Base_network, default_settings
from run_settings import Net_settings
import Weightstore as ws

class simple_model(Base_network):

    def __init__(self,settings):

        Base_network.__init__(self,settings)



    def get_model_test(self):

        model = self.load_model()
        if model is None:
            model = self.get_model()

        model.compile(loss='binary_crossentropy',
                      optimizer='rmsprop',
                      metrics=['accuracy','recall','precision'])


        model.load_weights(self.settings.save_weights_path)

        return model

    def get_model(self):

        model = Sequential()

        model.add(Convolution2D(10, 7, 7, activation='relu', border_mode='same',input_shape=(3, self.settings.img_width, self.settings.img_height), init='glorot_uniform'))
        model.add(Dropout(0.5))
        model.add(MaxPooling2D(pool_size=(2, 2)))


        model.add(Convolution2D(10, 7, 7, activation='relu', border_mode='same', init='glorot_uniform'))
        model.add(Dropout(0.5))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Convolution2D(10, 7, 7, activation='relu', border_mode='same', init='glorot_uniform'))
        model.add(Dropout(0.5))

        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Flatten())
        model.add(Dense(256, activation='relu', init='glorot_uniform'))
        model.add(Dropout(0.5))
        model.add(Dense(256, activation='relu', init='glorot_uniform'))
        model.add(Dropout(0.5))
        model.add(Dense(256, activation='relu', init='glorot_uniform'))
        model.add(Dropout(0.5))
        model.add(Dense(1, activation='sigmoid'))

        print self.has_weights()

        if self.has_weights():
            model.load_weights(self.settings.save_weights_path)
            print "loaded_model"
        elif False: # setting of pretraining turned off
            print "No Weights found settings pretrained weights"
            model = self.set_pretrained_weights(model)

        return model


    def get_model_train(self):
        model = self.get_model()
        # compile the model with a SGD/momentum optimizer
        # and a very slow learning rate.
        model.compile(loss='binary_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])
        return model

    def model_name(self):
        return "simple_model"




def train(guid_substring = None):
    settings = ws.get_settings(guid_substring)
    if settings == None:
        settings = default_settings()

    net = simple_model(settings)
    net.fine_tune_and_save()


def train_dataset(dataset, guid_substring = None):
    settings = ws.get_settings(guid_substring)
    if settings == None:
        settings = default_settings(dataset)


    net = simple_model(settings)
    net.fine_tune_and_save()


def get_model_test(settings):
    net = simple_model(settings)
    return net.get_model_test()

def get_net(settings):
    return simple_model(settings)


if __name__ == "__main__":
    sys.argv = filter(lambda x : x != '',sys.argv )
    guid_substring = sys.argv[-1]
    train(guid_substring)
