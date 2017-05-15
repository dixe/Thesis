import keras
import sys

class EarlyStoppingByLossVal(Callback):
    def __init__(self, monitor='loss', value=0.01, verbose=0):
        super(Callback, self).__init__()
        self.monitor = monitor
        self.value = value
        self.verbose = verbose

    def on_epoch_end(self, epoch, logs={}):
        current = logs.get(self.monitor)
        if current is None:
            print("Early stopping requires %s available!" % self.monitor)

        elif current < self.value:
            if self.verbose > 0:
                print("Epoch %05d: early stopping THR" % epoch)
            self.model.stop_training = True

class Stop_callback(keras.callbacks.Callback):


    def on_epoch_end(self, epoch, logs):
        print sys.stdin
        if not sys.stdin.isatty():
            print "Not empty"
            exit()
