import dynet_config
dynet_config.set(autobatch=1, profiling=0)
#dynet_config.set_gpu(False)
import dynet as dy
import numpy as np
import array

class DynetModel:
    """
    The sequence tagger model, dynet implementation.
    """
    def __init__(self, embedding, output_size, hidden_size, seed=1, crf=False, dropout_rate=0.5):
        self.set_seed(seed)
        self.model = dy.ParameterCollection()
        self.trainer = dy.SimpleSGDTrainer(self.model)

        # CRF
        if crf:
            self.num_tags = output_size + 2  # Add 2 to account for start and end tags in CRF
            self.trans_mat = self.model.add_parameters((self.num_tags, self.num_tags))
            self._loss = self._calculate_crf_loss
            self._predict = self._crf_predict_sentence
        else:
            self.num_tags = output_size
            self._loss = self._calculate_loss
            self._predict = self._predict_sentence

        # Embedding
        self.lookup = self.model.lookup_parameters_from_numpy(embedding.vectors)
        (embed_size, _), _ = self.lookup.dim()

        # Bi-LSTM
        self.bilstm = dy.BiRNNBuilder(
                            num_layers = 1,
                            input_dim = embed_size,
                            hidden_dim = hidden_size * 2,
                            model = self.model,
                            rnn_builder_factory = dy.LSTMBuilder)

        # Dense layer
        self.w = self.model.add_parameters((self.num_tags, hidden_size * 2))
        self.b = self.model.add_parameters(self.num_tags)

        self.dropout_rate = dropout_rate

    def set_seed(self, seed):
        dy.reset_random_seed(seed)
        np.random.seed(seed)
       
    def _calculate_score(self, sentence):
        """
        Calculates a score distribution for each word in a sentence.
        Runs through embeddding, bi-lstm, and a linear layer.
        """
        embeddings = [self.lookup[w] for w in sentence]
        bi_lstm_output = self.bilstm.transduce(embeddings)
        return [self.w * o + self.b for o in bi_lstm_output]

    def _calculate_train_score(self, sentence):
        """Same as _calculate_score, but applies dropout after embedding and bi-lstm layers, used for training"""
        embeddings = [self.lookup[w] for w in sentence]
        embeddings = [dy.dropout(e, self.dropout_rate) for e in embeddings]
        bi_lstm_output = self.bilstm.transduce(embeddings)
        bi_lstm_output = [dy.dropout(o, self.dropout_rate) for o in bi_lstm_output]
        return [self.w * o + self.b for o in bi_lstm_output]

    def _calculate_crf_loss(self, sentence, labels):
        labels = [l + 2 for l in labels] # Transformation to account for start and end tag in CRF
        score_vecs = self._calculate_train_score(sentence) 
        instance_score = self._forward(score_vecs)
        gold_tag_indices = array.array('I', labels)
        gold_score = self._score_sentence(score_vecs, gold_tag_indices)
        return instance_score - gold_score

    def _calculate_loss(self, sentence, labels):
        score_vecs = self._calculate_train_score(sentence)
        probs = [dy.softmax(z) for z in score_vecs]
        losses = [-dy.log(dy.pick(dist, label)) for dist, label in zip(probs, labels)]
        return dy.esum(losses)

    def fit_auto_batch(self, sentences, labels, mini_batch_size = 1, epochs = 1, 
                       patience = None, validation_sentences = None, validation_labels = None):
        """
        Train the model using dynet' auto-batching (requires --dynet-autobatch 1). 
        The model expects the sentence and labels transformed into integer representations.
        """
        if patience is not None:
            if validation_sentences is None or validation_labels is None:
                raise ValueError("Patience is set but no validation sentences or labels")
            best_correct = 0

        train_pairs = list(zip(sentences, labels))

        for epoch in range(epochs):
            np.random.shuffle(train_pairs)
            mini_batches = [train_pairs[x:x+mini_batch_size] for x in range(0, len(train_pairs), mini_batch_size)]

            for batch in mini_batches:
                dy.renew_cg()
                losses = []
                for x, y in batch:
                    loss = self._loss(x, y)
                    losses.append(loss)
                loss = dy.esum(losses)
                loss.forward()
                loss.backward()
                self.trainer.update()

            if patience is not None:
                correct = self.evaluate(validation_sentences, validation_labels)
                if correct > best_correct:
                    best_correct = correct
                    epochs_no_improvement = 0
                    self.save("tmp_patience_model.model")
                else:
                    epochs_no_improvement += 1
                    if patience == epochs_no_improvement:
                        self.load("tmp_patience_model.model")
                        return epoch + 1
                
        return epochs

    def evaluate(self, sentences, labels):
        correct = 0
        for sentence, sentence_labels in zip(sentences, labels):
            predictions = self.predict(sentence)
            for prediction, label in zip(predictions, sentence_labels):
                if prediction == label: correct += 1
        return correct

    def predict_batch(self, sentences):
        raise NotImplemented("TODO")

    def predict(self, sentence):
        """
        Predicts the tags of a sentence. 
        The model expects the sentence transformed into integer representations.
        """
        return self._predict(sentence)

    def _predict_sentence(self, sentence):
        dy.renew_cg()
        score_vecs = self._calculate_score(sentence)
        probs = [dy.softmax(z) for z in score_vecs]
        return [np.argmax(dist.value()) for dist in probs]

    def _crf_predict_sentence(self, sentence):
        dy.renew_cg()
        score_vecs = self._calculate_score(sentence)
        pred_tag_indices, _  = self._viterbi(score_vecs)
        return [i - 2 for i in pred_tag_indices] # Subtract 2 to account for start and end tag in CRF

    # code based on https://github.com/rguthrie3/BiLSTM-CRF
    def _viterbi(self, observations, unk_tag=None, dictionary=None):
        START_TAG = 0
        END_TAG = 1
        backpointers = []
        init_vvars   = [-1e10] * self.num_tags
        init_vvars[START_TAG] = 0 # <Start> has all the probability
        for_expr     = dy.inputVector(init_vvars)
        trans_exprs  = [self.trans_mat[idx] for idx in range(self.num_tags)]
        for obs in observations:
            bptrs_t = []
            vvars_t = []
            for next_tag in range(self.num_tags):
                next_tag_expr = for_expr + trans_exprs[next_tag]
                next_tag_arr = next_tag_expr.npvalue()
                best_tag_id  = np.argmax(next_tag_arr)
                if unk_tag:
                    best_tag = self.index2tag[best_tag_id]
                    if best_tag == unk_tag:
                        next_tag_arr[np.argmax(next_tag_arr)] = 0 # set to 0
                        best_tag_id = np.argmax(next_tag_arr) # get second best

                bptrs_t.append(best_tag_id)
                vvars_t.append(dy.pick(next_tag_expr, best_tag_id))
            for_expr = dy.concatenate(vvars_t) + obs
            backpointers.append(bptrs_t)
        # Perform final transition to terminal
        terminal_expr = for_expr + trans_exprs[END_TAG]
        terminal_arr  = terminal_expr.npvalue()
        best_tag_id   = np.argmax(terminal_arr)
        path_score    = dy.pick(terminal_expr, best_tag_id)
        # Reverse over the backpointers to get the best path
        best_path = [best_tag_id] # Start with the tag that was best for terminal
        for bptrs_t in reversed(backpointers):
            best_tag_id = bptrs_t[best_tag_id]
            best_path.append(best_tag_id)
        start = best_path.pop() # Remove the start symbol
        best_path.reverse()
        assert start == START_TAG
        # Return best path and best path's score
        return best_path, path_score

    # code adapted from K.Stratos' code basis (https://github.com/bplank/bilstm-aux/blob/master/src/lib/mnnl.py)  
    def _score_sentence(self, score_vecs, tags):
        START_TAG = 0
        END_TAG = 1
        assert(len(score_vecs)==len(tags))
        tags.insert(0, START_TAG) # add start
        total = dy.scalarInput(.0)
        for i, obs in enumerate(score_vecs):
            # transition to next from i and emission
            next_tag = tags[i + 1]
            total += dy.pick(self.trans_mat[next_tag], tags[i]) + dy.pick(obs, next_tag)
        total += dy.pick(self.trans_mat[END_TAG],tags[-1])
        return total

    # code based on https://github.com/bplank/bilstm-aux/blob/master/src/lib/mnnl.py
    def _forward(self, observations):
        START_TAG = 0
        END_TAG = 1
        # calculate forward pass
        def log_sum_exp(scores):
            npval = scores.npvalue()
            argmax_score = np.argmax(npval)
            max_score_expr = dy.pick(scores, argmax_score)
            max_score_expr_broadcast = dy.concatenate([max_score_expr] * self.num_tags)
            return max_score_expr + dy.logsumexp_dim((scores - max_score_expr_broadcast), 0)

        init_alphas = [-1e10] * self.num_tags
        init_alphas[START_TAG] = 0
        for_expr = dy.inputVector(init_alphas)
        for obs in observations:
            alphas_t = []
            for next_tag in range(self.num_tags):
                obs_broadcast = dy.concatenate([dy.pick(obs, next_tag)] * self.num_tags)
                next_tag_expr = for_expr + self.trans_mat[next_tag] + obs_broadcast
                alphas_t.append(log_sum_exp(next_tag_expr))
            for_expr = dy.concatenate(alphas_t)
        terminal_expr = for_expr + self.trans_mat[END_TAG]
        alpha = log_sum_exp(terminal_expr)
        return alpha

    def save(self, filepath):
        self.model.save(filepath)

    def load(self, filepath):
        self.model.populate(filepath)
