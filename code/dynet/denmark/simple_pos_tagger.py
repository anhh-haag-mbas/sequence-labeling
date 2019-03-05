import numpy as np
import dynet as dy
import ipdb
from random import shuffle

class SimplePOSTagger:
    def __init__(self, vocab_size, output_size, embed_size = 86, hidden_size = 16):
        self.model = dy.ParameterCollection()
        self.trainer = dy.SimpleSGDTrainer(self.model)

        self.lookup = self.model.add_lookup_parameters((vocab_size, embed_size))
        self.lstm = dy.LSTMBuilder(
                        layers = 1,
                        input_dim = embed_size,
                        hidden_dim = hidden_size,
                        model = self.model)

        self.w = self.model.add_parameters((output_size, hidden_size))
        self.b = self.model.add_parameters(output_size)

    def single_sentence_loss(self, sentence, labels):
        dy.renew_cg()
        s = self.lstm.initial_state()
        loss = []

        for word, label in zip(sentence, labels):
            embedding = self.lookup[word]
            s = s.add_input(embedding)

            prob_dist = dy.softmax(self.w * s.output() + self.b)
            loss.append(-dy.log(dy.pick(prob_dist, label)))

        return dy.esum(loss)

    def fit(self, inputs, labels, mini_batch_size = 1, epochs = 1):
        """
        Expects the inputs and labels to be transformed to integers beforehand
        """
        for sentence, sentence_labels in zip(inputs, labels):
            dy.renew_cg()
            inps = [self.lookup[i] for i in sentence]
            s = self.lstm.initial_state()
            outs = s.transduce(inps)

            probs = [dy.softmax(self.w * out + self.b) for out in outs]
            losses = [-dy.log(dy.pick(dist, label)) for dist, label in zip(probs, sentence_labels)]

            loss = dy.esum(losses)
            loss.value()
            loss.backward()
            self.trainer.update()


    def fit_auto_batch(self, inputs, labels, mini_batch_size = 1, epochs = 1):
        for epoch in range(epochs):
            dy.renew_cg()
            losses = []


#    def fit_auto_batch(self, inputs, labels, mini_batch_size = 1, epochs = 1):
#        training_pairs = list(zip(inputs, labels))
#        data = [[] for _ in range(epochs)]
#        for epoch in range(epochs):
#            dy.renew_cg()
#            shuffle(training_pairs) 
#            print(f"epochs {epochs}, epoch {epoch}, mini_batch_size {mini_batch_size}, iterations {len(training_pairs)/mini_batch_size}")
#            for i in range(int(len(training_pairs) / mini_batch_size)):
#                losses = []
#                for b in range(mini_batch_size):
#                    sentence, sentence_labels = training_pairs[b + i * mini_batch_size]
#                    loss = self.single_sentence_loss(sentence, sentence_labels)
#                    losses.append(loss)
#                batch_loss = dy.esum(losses) / mini_batch_size
#                loss_value = batch_loss.value()
#                if (i % 100 == 0): 
#                    print(loss_value)
#                data[epoch].append(loss_value)
#                batch_loss.backward()
#                self.trainer.update()
#
#        return data
#
    def fit_batch(self, inputs, labels, mini_batch_size = 1, epochs = 1):
        pass

    def predict(self, sentence):
        dy.renew_cg()
       
        s = self.lstm.initial_state()
        probs = []
        for word in sentence:
            embedding = self.lookup[word]
            s = s.add_input(embedding)
            prob_dist = dy.softmax(self.w * s.output() + self.b)
            probs.append(prob_dist)

        dy.forward(probs)
           
        return [np.argmax(dist.value()) for dist in probs]

    def predict_auto_batch(self, sentences):
        dy.renew_cg()

        distributions = []
         
        for sentence in sentences:
            s = self.lstm.initial_state()
            sentence_probs = []
            for word in sentence:
                embedding = self.lookup[word]
                s = s.add_input(embedding)
                prob_dist = dy.softmax(self.w * s.output() + self.b)
                sentence_probs.append(prob_dist)

            distributions.append(sentence_probs)

        dy.forward(distributions)

        return "Not done"

    def predict_batch(self, sentences):
        pass

    def save(self, filepath):
        self.model.save(filepath)

    def load(self, filepath):
        self.model.populate(filepath)

