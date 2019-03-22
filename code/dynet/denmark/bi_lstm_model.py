import dynet as dy
import numpy as np
import ipdb
import array
import random

class BiLstmModel:

    def __init__(self, vocab_size, output_size, embed_size = 86, hidden_size = 8, embeddings = None):
        self.name = self.__class__.__name__
        self.model = dy.ParameterCollection()
        self.trainer = dy.SimpleSGDTrainer(self.model)

        # Embedding
        if embeddings is None:
            self.lookup = self.model.add_lookup_parameters((vocab_size, embed_size))
        else:
            self.lookup = self.model.lookup_parameters_from_numpy(embeddings)
            (embed_size, vocab_size), _ = self.lookup.dim()


        self.bilstm = dy.BiRNNBuilder(
                        num_layers = 1,
                        input_dim = embed_size,
                        hidden_dim = hidden_size * 2,
                        model = self.model,
                        rnn_builder_factory = dy.LSTMBuilder)

        # Dense layer
        self.w = self.model.add_parameters((output_size, hidden_size * 2))
        self.b = self.model.add_parameters(output_size)
        
    def _calculate_loss(self, sentence, sentence_labels):
        # Embedding + Bi-LSTM + Linear layer
        embeddings = [self.lookup[w] for w in sentence]
        bilstm_output = self.bilstm.transduce(embeddings)
        probs = [dy.softmax(self.w * o + self.b) for o in bilstm_output]
        losses = [-dy.log(dy.pick(dist, label)) for dist, label in zip(probs, sentence_labels)]
        return dy.esum(losses)

    def fit(self, data, labels):
        """
        Expects the inputs and labels to be transformed to integers beforehand
        """
        for sentence, sentence_labels in zip(data, labels):
            dy.renew_cg()
            loss = self._calculate_loss(sentence, sentence_labels)

            loss.value()
            loss.backward()
            self.trainer.update()

    def fit_auto_batch(self, data, labels, mini_batch_size = 1, epochs = 1):
        train_pairs = list(zip(data, labels))
        loss_progression = []

        for epoch in range(epochs):
            random.shuffle(train_pairs)
            mini_batches = [train_pairs[x:x+mini_batch_size] for x in range(0, len(train_pairs), mini_batch_size)]

            for batch in mini_batches:
                dy.renew_cg()
                losses = []
                for sentence, sentence_labels in batch:
                    loss = self._calculate_loss(sentence, sentence_labels)
                    losses.append(loss)
                loss = dy.esum(losses)

                loss_value = loss.value()
                loss.backward()
                self.trainer.update()

                loss_progression.append(loss_value)

        return loss_progression

    def fit_batch(self, inputs, labels, mini_batch_size = 1, epochs = 1):
        pass

    def predict(self, sentence):
        dy.renew_cg()
         # Embedding + Bi-LSTM + Linear layer
        embeddings = [self.lookup[w] for w in sentence]
        bi_lstm_output = self.bilstm.transduce(embeddings)
        score_vecs = [dy.softmax(self.w * o + self.b) for o in bi_lstm_output]
        return [np.argmax(props.value()) for props in score_vecs]

    def predict_auto_batch(self, sentences):
        pass

    def predict_batch(self, sentences):
        pass

    def save(self, filepath):
        self.model.save(filepath)

    def load(self, filepath):
        self.model.populate(filepath)

