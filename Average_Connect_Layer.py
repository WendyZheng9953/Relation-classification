import tensorflow as tf

from tensorflow.keras import backend as K
from tensorflow.keras.layers import Layer
from tensorflow.keras.layers import InputSpec, Dense, concatenate, Dropout


class AverageConnectLayer(Layer):

    def __init__(self, **kwargs):
        self.init = tf.initializers.random_uniform()
        self.supports_masking = True
        super(AverageConnectLayer, self).__init__(**kwargs)

    def compute_mask(self, input, input_mask=None):
        if isinstance(input_mask, list):
            return [None] * len(input_mask)
        else:
            return None

    def build(self, input_shape):
        self.bert_input_shape = (None, None, 768)
        self.entity1_shape = (None, 128)
        self.entity2_shape = (None, 128)
        self.input_spec = [InputSpec(ndim=3)]

        super(AverageConnectLayer, self).build([(None, None, 768), (None, 128), (None, 128)])

    def call(self, inputs, mask=None):
        bert_input, entity1_input, entity2_input = inputs

        # [CLS] bert_output
        h0 = K.tanh(bert_input[:, 0, :])
        # Entity one and two output
        h1 = K.sum(bert_input * K.expand_dims(entity1_input, -1), axis=1, keepdims=False)
        h2 = K.sum(bert_input * K.expand_dims(entity2_input, -1), axis=1, keepdims=False)
        h1 = K.tanh(h1 / K.sum(entity1_input, axis=1, keepdims=True))
        h2 = K.tanh(h2 / K.sum(entity2_input, axis=1, keepdims=True))

        # add dropout
        h0 = Dropout(0.1)(h0)
        h1 = Dropout(0.1)(h1)
        h2 = Dropout(0.1)(h2)

        # fully connected layer for each vector
        h0 = Dense(self.bert_input_shape[2])(h0)
        h1 = Dense(self.bert_input_shape[2])(h1)
        h2 = Dense(self.bert_input_shape[2])(h2)

        return concatenate([h0, h1, h2], -1)

    def get_output_shape_for(self, input_shape):
        return self.compute_output_shape(input_shape)

    def compute_output_shape(self, input_shape):
        bert_input_shape, mark_input_shape = input_shape

        return (bert_input_shape[0], bert_input_shape[2] * 3)

