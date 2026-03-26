from tensorflow.keras.layers import Layer
from tensorflow.keras import backend as K
import tensorflow as tf

class Attention(Layer):
    def __init__(self, return_sequences=True, **kwargs):
        super(Attention, self).__init__(**kwargs)
        self.return_sequences = return_sequences

    def build(self, input_shape):
        # input_shape = (batch_size, timesteps, features)
        self.W = self.add_weight(name="att_weight",
                                 shape=(input_shape[-1], 1),
                                 initializer="normal",
                                 trainable=True)
        self.b = self.add_weight(name="att_bias",
                                 shape=(1,),  # make bias scalar for broadcasting
                                 initializer="zeros",
                                 trainable=True)
        super(Attention, self).build(input_shape)

    def call(self, x):
        # x shape: (batch, timesteps, features)
        e = K.tanh(K.dot(x, self.W) + self.b)  # shape: (batch, timesteps, 1)
        a = K.softmax(e, axis=1)               # attention weights
        output = x * a                          # broadcasting works
        if self.return_sequences:
            return output
        return K.sum(output, axis=1)           # sum over timesteps