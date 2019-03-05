import dynet as dy
import numpy as np
import ipdb

class CrfBiPosTaggerDouble:
    def __init__(self, vocab_size, output_size, embed_size = 86, hidden_size = 8):
        self.model = dy.ParameterCollection()
        self.trainer = dy.SimpleSGDTrainer(self.model)

        self.lookup = self.model.add_lookup_parameters((vocab_size, embed_size))
        self.lstmf = dy.LSTMBuilder(
                        layers = 1,
                        input_dim = embed_size,
                        hidden_dim = hidden_size,
                        model = self.model)
        
        self.lstmb = dy.LSTMBuilder(
                        layers = 1,
                        input_dim = embed_size,
                        hidden_dim = hidden_size,
                        model = self.model)


        self.w = self.model.add_parameters((output_size, hidden_size * 2))
        self.b = self.model.add_parameters(output_size)

    def fit(self, inputs, labels, mini_batch_size = 1, epochs = 1):
        """
        Expects the inputs and labels to be transformed to integers beforehand
        """
        for sentence, sentence_labels in zip(inputs, labels):
            dy.renew_cg()
            inps = [self.lookup[i] for i in sentence]

            sfs = [self.lstmf.initial_state()]
            sbs = [self.lstmb.initial_state()]

            for i in range(len(inps)):
                sf = sfs[i].add_input(inps[i])
                sb = sbs[i].add_input(inps[-(i + 1)])

                sfs.append(sf)
                sbs.append(sb)

            sfs = sfs[1:]
            sbs = sbs[1:]
            sbs = sbs[::-1]

            probs = [dy.softmax(self.w * dy.concatenate([sf.output(),sb.output()]) + self.b) for sf, sb in zip(sfs, sbs)]
            losses = [-dy.log(dy.pick(dist, label)) for dist, label in zip(probs, sentence_labels)]

            loss = dy.esum(losses)
            loss.value()
            loss.backward()
            self.trainer.update()


    def fit_batch(self, inputs, labels, mini_batch_size = 1, epochs = 1):
        pass

    def fit_auto_batch(self, inputs, labels, mini_batch_size = 1, epochs = 1):
        pass

    def predict(self, sentence):
        dy.renew_cg()
        inps = [self.lookup[i] for i in sentence]

        sfs = [self.lstmf.initial_state()]
        sbs = [self.lstmb.initial_state()]

        for i in range(len(inps)):
            sf = sfs[i].add_input(inps[i])
            sb = sbs[i].add_input(inps[-(i + 1)])

            sfs.append(sf)
            sbs.append(sb)

        sfs = sfs[1:]    # Remove initial state
        sbs = sbs[:0:-1] # Reverse and remove initial state

        probs = [dy.softmax(self.w * dy.concatenate([sf.output(),sb.output()]) + self.b) for sf, sb in zip(sfs, sbs)]

        return [np.argmax(dist.value()) for dist in probs]

    def predict_auto_batch(self, sentences):
        pass

    def predict_batch(self, sentences):
        pass

    def save(self, filepath):
        self.model.save(filepath)

    def load(self, filepath):
        self.model.populate(filepath)

