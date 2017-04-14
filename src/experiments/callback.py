import keras
import sys


class Stop_callback(keras.callbacks.Callback):


    def on_epoch_end(self, epoch, logs):
        print sys.stdin
        if not sys.stdin.isatty():
            print "Not empty"
            exit()
