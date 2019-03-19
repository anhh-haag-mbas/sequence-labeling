import dynet as dy
import numpy as np
import ipdb
import array

START_TAG = 0
END_TAG = 1

class CrfBiPosTaggerDouble:
    def __init__(self, vocab_size, output_size, embed_size = 86, hidden_size = 8):
        self.model = dy.ParameterCollection()
        self.trainer = dy.SimpleSGDTrainer(self.model)

        # Embedding
        self.lookup = self.model.add_lookup_parameters((vocab_size, embed_size))

        # Bi-LSTM
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
        for sentence, sentence_labels in list(zip(inputs, labels)):
            dy.renew_cg()
            sentence_labels = [l for l in sentence_labels] # Transform(+2) to account for start and end tag
            inps = [self.lookup[i] for i in sentence]
           
            # LSTM
            f_init = self.lstmf.initial_state()
            b_init = self.lstmb.initial_state()
            
            forward = f_init.transduce(inps)
            backward = b_init.transduce(reversed(inps))

            concat_layer = [dy.concatenate([f,b]) for f, b in zip(forward, reversed(backward))]

            # Linear layer
            score_vecs = [(self.w * layer + self.b) for layer in concat_layer]

            # CRF
            #_, instance_score  = self.viterbi(score_vecs)
            instance_score = self.forward(score_vecs)

            gold_tag_indices = array.array('I', sentence_labels)

            gold_score = self.score_sentence(score_vecs, gold_tag_indices)

            loss = instance_score - gold_score

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
       
        # LSTM
        f_init = self.lstmf.initial_state()
        b_init = self.lstmb.initial_state()
        
        forward = f_init.transduce(inps)
        backward = b_init.transduce(reversed(inps))

        concat_layer = [dy.concatenate([f,b]) for f, b in zip(forward, reversed(backward))]

        # Linear layer
        score_vecs = [dy.rectify(self.w * layer + self.b) for layer in concat_layer]

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

