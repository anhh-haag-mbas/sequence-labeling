import dynet as dy
import numpy as np
import array

START_TAG = 0
END_TAG = 1

class BiLstmCrfModel:
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

        # Bi-LSTM
        self.bilstm = dy.BiRNNBuilder(
                            num_layers = 1, 
                            input_dim = embed_size,
                            hidden_dim = hidden_size * 2,
                            model = self.model,
                            rnn_builder_factory = dy.LSTMBuilder)

        self.num_tags = output_size
        # Dense layer
        self.w = self.model.add_parameters((self.num_tags, hidden_size * 2))
        self.b = self.model.add_parameters(self.num_tags)
        # For CRF
        self.trans_mat = self.model.add_parameters((self.num_tags, self.num_tags))
        
    def fit(self, inputs, labels, mini_batch_size = 1, epochs = 1):
        """
        Expects the inputs and labels to be transformed to integers beforehand
        """
        for sentence, sentence_labels in zip(inputs, labels):
            dy.renew_cg()
            loss = self._calculate_loss(sentence, sentence_labels)

            loss.value()
            loss.backward()
            self.trainer.update()

    def _calculate_loss(self, sentence, labels):
        # Embedding + Bi-LSTM + Linear layer
        embeddings = [self.lookup[w] for w in sentence]
        bi_lstm_output = self.bilstm.transduce(embeddings)
        score_vecs = [self.w * o + self.b for o in bi_lstm_output]

        # CRF
        instance_score = self.forward(score_vecs)
        gold_tag_indices = array.array('I', labels)
        gold_score = self.score_sentence(score_vecs, gold_tag_indices)
        return instance_score - gold_score

    def fit_batch(self, data, labels, mini_batch_size = 1, epochs = 1):
        train_pairs = list(zip(data, labels))

        for epoch in range(epochs):
            np.random.shuffle(train_pairs)
            mini_batches = [train_pairs[x:x+mini_batch_size] for x in range(0, len(train_pairs), mini_batch_size)]

            for batch in mini_batches:
                dy.renew_cg()
                losses = []
                for x, y in batch:
                    loss = self._calculate_loss(x,y)
                    losses.append(loss)
                loss = dy.esum(losses)
                loss.forward()
                loss.backward()
                self.trainer.update()

    def fit_auto_batch(self, inputs, labels, mini_batch_size = 1, epochs = 1):
        pass

    def predict(self, sentence):
        dy.renew_cg()
        inps = [self.lookup[i] for i in sentence]
       
        # LSTM
        bi_lstm_output = self.bilstm.transduce(inps)

        # Linear layer
        score_vecs = [dy.rectify(self.w * o + self.b) for o in bi_lstm_output]

        # CRF
        pred_tag_indices, _  = self.viterbi(score_vecs)
        return [i for i in pred_tag_indices]

    # code based on https://github.com/rguthrie3/BiLSTM-CRF
    def viterbi(self, observations, unk_tag=None, dictionary=None):
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
    def score_sentence(self, score_vecs, tags):
        assert(len(score_vecs)==len(tags))
        tags.insert(0, START_TAG) # add start
        total = dy.scalarInput(.0)
        for i, obs in enumerate(score_vecs):
            # transition to next from i and emission
            next_tag = tags[i + 1]
            total += dy.pick(self.trans_mat[next_tag],tags[i]) + dy.pick(obs,next_tag)
        total += dy.pick(self.trans_mat[END_TAG],tags[-1])
        return total

    # https://github.com/bplank/bilstm-aux/blob/master/src/lib/mnnl.py
    def forward(self, observations):
        # calculate forward pass
        def log_sum_exp(scores):
            npval = scores.npvalue()
            argmax_score = np.argmax(npval)
            max_score_expr = dy.pick(scores, argmax_score)
            max_score_expr_broadcast = dy.concatenate([max_score_expr] * self.num_tags)
            return max_score_expr + dy.logsumexp_dim((scores - max_score_expr_broadcast),0)

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


    def predict_auto_batch(self, sentences):
        pass

    def predict_batch(self, sentences):
        pass

    def save(self, filepath):
        self.model.save(filepath)

    def load(self, filepath):
        self.model.populate(filepath)

