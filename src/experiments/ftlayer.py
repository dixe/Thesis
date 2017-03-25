"""
A simple flow trought layer for just passing input trough
"""

from keras import backend as K
from keras.engine.topology import Layer


class FTLayer(Layer):

    def __init__(self, **kwargs):
        super(FTLayer,self).__init__(**kwargs)

    def build(self, input_shape):
        super(FTLayer,self).build(input_shape)

    def call(self, x, **kwargs):
        return x
