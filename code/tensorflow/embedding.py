import tensorflow as tf
from tensorflow.contrib.rnn import LSTMCell as LstmCell
from tensorflow.nn import embedding_lookup
from tensorflow.layers import Layer


class Embedding(Layer):
    def __init__(self, input_dimensions, output_dimensions, **kwargs):
        super(Embedding, self).__init__(**kwargs)

        # The amount of words in our vocabulary
        self.vocab_size = input_dimensions
        # When embedding a how, how many dimensions do the embedded vector have
        self.embedding_size = output_dimensions

    def build(self, input_shape):
        self.embedding = self.add_variable("embedding", shape=[self.vocab_size, self.embedding_size])

    def call(self, input):
        return embedding_lookup(self.embedding, input)
