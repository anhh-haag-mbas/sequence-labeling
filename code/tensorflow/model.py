import time

import keras
from keras.layers import Dense, LSTM, Activation, Embedding, Bidirectional
from keras.models import Sequential
from keras.optimizers import SGD
from polyglot.downloader import downloader

from sentences import Sentences

# Configuration dict
#
# framework: "tensorflow", "dynet", "pytorch"
# model: "bi-lstm", "bi-lstm+crf"
# language: "da", "de"
# optimizer: "sgd", "adams"
# learning rate: 0.01, 0.1
# embedding type: "task specific", "fasttext", "polyglot"
# embedding dimensions: int (this only applies to "embedding type" = "task specific")
# dropout: float
# task: "pos", "ner"
# batch_size: int

# Result dict
#
# configuration: the configuration used to produce this result
# model: framework specific trained model
# training_time: seconds it took to train
# epochs: epochs trained
# evaluation_time: seconds it took to evaluate
# accuracy_total: int
# accuracy_oov: int

example_config = {
    "framework": "tensorflow",
    "model": "bi-lstm",
    "language": "da",
    "optimizer": "sgd",
    "learning rate": 0.1,
    "embedding type": "task specific",
    "embedding dimensions": 64,
    "dropout": 0,
    "task": "pos",
    "batch_size": 1,
    "repeat": 2
}


class TensorFlowSequenceLabelling:
    def __init__(self, configuration):
        self.configuration = configuration

    def load_sentences(self):
        self.sentences = Sentences(task=self.configuration["task"], language_code=self.configuration["language"])

    def load_embeddings(self):
        if self.configuration["embedding type"] == "polyglot":
            downloader.download(f"embeddings2.{self.configuration['language']}")

    def create_model(self):
        self.model = Sequential()

        sentence_length = self.sentences.sentence_length
        embedding_dimensions = self.configuration["embedding dimensions"]
        self.model.add(Embedding(self.sentences.word_count, embedding_dimensions, input_length=sentence_length,
                                 mask_zero=True))

        # Bidirectional returns the hidden size*2 as there are two layers now (one in each direction)
        self.model.add(Bidirectional(LSTM(units=100, return_sequences=True)))

        self.model.add(Dense(self.sentences.tag_count))
        self.model.add(Activation('softmax'))

        self.model.compile(loss='categorical_crossentropy',
                           optimizer=self.optimizer(),
                           metrics=['accuracy'])

        self.model.summary()
        print("words", self.sentences.training_word_ids.shape)
        print("tags", self.sentences.training_tag_ids.shape)

    def optimizer(self):
        if self.configuration["optimizer"] == "sgd":
            return SGD(lr=self.configuration["learning rate"])
        if self.configuration["optimizer"] == "adam":
            return "adam"

    def train_model(self):
        # convergence = EarlyStopping(monitor='val_loss', min_delta=0, patience=3, restore_best_weights=True)
        #
        # history = model.fit(sentences.training_word_ids, keras.utils.to_categorical(sentences.training_tag_ids),
        #                     validation_data=(sentences.validation_word_ids, keras.utils.to_categorical(sentences.validation_tag_ids)),
        #                     batch_size=10, epochs=50, callbacks=[convergence])

        begin_timestamp = time.time()
        self.model.fit(self.sentences.training_word_ids,
                       keras.utils.to_categorical(self.sentences.training_tag_ids),
                       batch_size=self.configuration["batch_size"], epochs=1)
        end_timestamp = time.time()

        self.training_time = end_timestamp - begin_timestamp

    def evaluate_model(self):
        begin_timestamp = time.time()
        self.loss_total, self.accuracy_total = self.model.evaluate(self.sentences.testing_word_ids,
                                                                   keras.utils.to_categorical(
                                                                       self.sentences.testing_tag_ids))
        end_timestamp = time.time()

        self.evaluation_time = end_timestamp - begin_timestamp

    def run(self):
        self.load_sentences()
        self.load_embeddings()
        self.create_model()
        self.train_model()
        self.evaluate_model()

        self.epochs = None  # TODO: self.epochs
        self.accuracy_oov = None  # TODO: self.accuracy_oov

        return {
            "configuration": self.configuration,
            "model": self.model,
            "training_time": self.training_time,
            "epochs": self.epochs,
            "evaluation_time": self.evaluation_time,
            "accuracy_total": self.accuracy_total,
            "accuracy_oov": self.accuracy_oov,
        }
