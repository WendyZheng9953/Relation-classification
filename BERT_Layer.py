from tensorflow.keras import backend as K
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np


class BERTLayer(tf.keras.layers.Layer):
    def __init__(self, n_fine_tune_layers=10, **kwargs):
        self.n_fine_tune_layers = n_fine_tune_layers
        self.trainable = True
        self.output_size = 768
        super(BERTLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.bert = hub.Module(
            "bert_hub",
            trainable=self.trainable,
            name="{}_module".format(self.name)
        )
        trainable_vars = self.bert.variables

        # Remove unused layers
        trainable_vars = [var for var in trainable_vars if not ("/cls/" in var.name or 'pooler' in var.name)]

        # Select how many layers to fine tune
        if self.n_fine_tune_layers == -1:
            trainable_vars = []
        else:
            trainable_vars = trainable_vars[-self.n_fine_tune_layers:]

        # Add to trainable weights
        for var in trainable_vars:
            self._trainable_weights.append(var)

        # Add non-trainable weights
        for var in self.bert.variables:
            if var not in self._trainable_weights:
                self._non_trainable_weights.append(var)

        super(BERTLayer, self).build(input_shape)

    # Function of the layer
    def call(self, inputs):
        inputs = [K.cast(x, dtype="int32") for x in inputs]
        input_ids, input_mask, segment_ids = inputs
        bert_inputs = dict(
            input_ids=input_ids, input_mask=input_mask, segment_ids=segment_ids
        )

        # Use "pooled_output" for classification tasks on an entire sentence.
        # Use "sequence_outputs" for token-level output.
        result = self.bert(inputs=bert_inputs, signature="tokens", as_dict=True)
        return result['sequence_output']

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_size)

