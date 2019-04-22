import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import time

import keras
from keras import Model, Input
from keras.callbacks import EarlyStopping
from keras.layers import Dense, LSTM, Embedding, Bidirectional, Dropout
from keras.optimizers import SGD
from polyglot.mapping import Embedding as PolyglotEmbedding

from .crf import CRF
from .sentences import Sentences


# TODO: Seed
#  see https://keras.io/getting-started/faq/#how-can-i-obtain-reproducible-results-using-keras-during-development
#  and https://stackoverflow.com/a/38423188
# TODO: Shamelessly copy from https://github.com/guillaumegenthial/tf_ner/blob/master/models/lstm_crf/main.py

class TensorFlowSequenceLabelling:
    def __init__(self, configuration):
        self.c = configuration
        self.embedding = self.load_embedding()
        self.sentences = self.load_sentences()
        self.model = self.create_model()

    def load_embedding(self):
        path = os.path.join(self.c["data_root"], "embeddings", self.c["language"] + ".tar.bz2")
        return PolyglotEmbedding.load(path)

    def load_sentences(self):
        return Sentences(task=self.c["task"], language_code=self.c["language"],
                         id_by_word=self.embedding.vocabulary.word_id)

    def create_model(self):
        sentence_length = self.sentences.sentence_length
        dimensions = len(self.embedding.zero_vector())
        embeddings_initializer = keras.initializers.constant(self.embedding.vectors)

        inputs = Input(shape=(sentence_length,))

        layer = Embedding(len(self.embedding.words), dimensions, input_length=sentence_length, mask_zero=True,
                          embeddings_initializer=embeddings_initializer)(inputs)

        if self.c["dropout"] and self.c["dropout"] > 0:
            layer = Dropout(self.c["dropout"])(layer)

        # Bidirectional returns the hidden size*2 as there are two layers now (one in each direction)
        layer = Bidirectional(LSTM(units=100, return_sequences=True))(layer)

        if self.c["dropout"] and self.c["dropout"] > 0:
            layer = Dropout(self.c["dropout"])(layer)

        layer = Dense(self.sentences.tag_count, activation='softmax')(layer)

        unpadded_sentence_lengths = Input(shape=[1], dtype='int32')
        if self.c["crf"]:
            crf = CRF()
            layer = crf([layer, unpadded_sentence_lengths])
        else:
            crf = None

        optimizer = None
        if self.c["optimizer"] == "sgd":
            optimizer = SGD(lr=self.c["learning_rate"])
        if self.c["optimizer"] == "adam":
            optimizer = "adam"

        model = Model(inputs=[inputs, unpadded_sentence_lengths], outputs=layer)

        if crf is not None:
            model.compile(loss=crf.loss,
                          optimizer=optimizer,
                          metrics=['accuracy'])
        else:
            model.compile(loss='categorical_crossentropy',
                          optimizer=optimizer,
                          metrics=['accuracy'])

        return model

    def run(self):
        training_time = self.measure(self.train_model)
        evaluation_time = self.measure(self.predict)
        self.evaluate()

        return {
            "configuration": self.c,
            "model": self.model,
            "training_time": training_time,
            "evaluation_time": evaluation_time,
            "total_values": self.total_values,
            "total_errors": self.total_errors,
            "total_oov": self.total_oov,
            "total_oov_errors": self.total_oov_errors,
            "total_acc": self.total_acc,
            "oov_acc": self.oov_acc,
            "epochs_run": self.epochs_run,
            # "sentence_errors": self.sentence_errors,
            "evaluation_matrix": self.evalutation_matrix
        }

    def train_model(self):
        if self.c["patience"]:
            callbacks = [EarlyStopping(monitor='val_acc', patience=self.c["patience"], restore_best_weights=True)]
        else:
            callbacks = None

        history = self.model.fit([self.sentences.training_word_ids, self.sentences.training_lengths],
                                 keras.utils.to_categorical(self.sentences.training_tag_ids,
                                                            num_classes=self.sentences.tag_count),
                                 validation_data=[
                                     [self.sentences.validation_word_ids, self.sentences.validation_lengths],
                                     keras.utils.to_categorical(self.sentences.validation_tag_ids,
                                                                num_classes=self.sentences.tag_count)],
                                 batch_size=self.c["batch_size"], epochs=self.c["epochs"], callbacks=callbacks, verbose=2)
        self.epochs_run = len(history.epoch)

    def predict(self):
        self.predictions = self.model.predict([self.sentences.testing_word_ids, self.sentences.testing_lengths]) \
            .argmax(2)

    def evaluate(self):
        actual_tags = self.sentences.testing_tag_ids
        words = self.sentences.testing_words()

        self.total_values = 0
        self.total_errors = 0
        self.total_oov = 0
        self.total_oov_errors = 0
        self.evalutation_matrix = {}
        for predicted_tag in self.sentences.tag_by_id.keys():
            for actual_tag in self.sentences.tag_by_id.keys():
                if actual_tag == self.sentences.tag_padding_id:
                    continue
                predicted_tag_name = self.sentences.tag_by_id[predicted_tag]
                actual_tag_name = self.sentences.tag_by_id[actual_tag]
                self.evalutation_matrix[(predicted_tag_name, actual_tag_name)] = 0

        self.sentence_errors = []
        for predicted_sentences, actual_sentences, sentences in zip(self.predictions, actual_tags, words):
            errors_in_sentence = 0
            for predicted_tag, actual_tag, word in zip(predicted_sentences, actual_sentences, sentences):
                error = predicted_tag != actual_tag
                oov = self.embedding.get(word) is None

                if actual_tag == self.sentences.tag_padding_id:
                    continue

                self.total_values += 1
                if error:
                    self.total_errors += 1
                    errors_in_sentence += 1
                if oov:
                    self.total_oov += 1
                if oov and error:
                    self.total_oov_errors += 1
                predicted_tag_name = self.sentences.tag_by_id[predicted_tag]
                actual_tag_name = self.sentences.tag_by_id[actual_tag]
                if predicted_tag_name not in self.evalutation_matrix:
                    self.evalutation_matrix[predicted_tag_name] = {}
                if actual_tag_name not in self.evalutation_matrix[predicted_tag_name]
                    self.evalutation_matrix[predicted_tag_name][actual_tag_name] = 0
                self.evalutation_matrix[predicted_tag_name][actual_tag_name] += 1
            self.sentence_errors += [{"errors": errors_in_sentence,
                                      "predicted_sentence": predicted_sentences,
                                      "actual_sentence": actual_sentences}]

        self.sentence_errors.sort(key=lambda x: x["errors"], reverse=True)
        self.total_acc = 1 - (self.total_errors / self.total_values)
        self.oov_acc = 1 - (self.total_oov_errors / self.total_oov)

    def measure(self, function):
        begin_timestamp = time.time()
        function()
        end_timestamp = time.time()

        return end_timestamp - begin_timestamp
