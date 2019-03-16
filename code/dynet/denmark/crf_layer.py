import numpy as np

class CrfLayer:
    def __init__(self, num_labels):
        self.num_labels = num_labels
        self.transitions = np.random.rand(num_labels + 2, num_labels + 2)
        self.transitions[(0, -1), :] = -1000
        self.transitions[0, -2] = 0
        self.transitions[-1, -1] = 0
        print(self.transitions)

    # Implementation inspired by: https://createmomo.github.io/2017/11/24/CRF-Layer-on-the-Top-of-BiLSTM-6/ 
    def viterbi(self, probability_distributions):
        alpha = ([],[])
        path = []
        transitions = self.transitions[1:1 + self.num_labels, 1:1 + self.num_labels]

        previous = probability_distributions[0, :] 
        for i in range(len(probability_distributions)):
            obs = probability_distributions[i, :] 
            
            # 1. Expand the previous
            previous = np.resize(previous, (self.num_labels, len(previous)))
            previous = np.transpose(previous)

            # 2. Expand the obs
            obs = np.resize(obs, (self.num_labels, len(obs)))
        
            # 3. Sum previous, obs and transition scores
            scores = previous + obs + transitions 

            # Change previous for next iteration 
            previous = np.amax(scores, axis = 0)

            # Add best scores and column indexes into alpha
            alpha[0].append(previous)
            alpha[1].append(scores.argmax(axis = 0)) # Index of highest value in each column
 
        path = []

        # Find the best path by moving backwards 
        idx = np.argmax(alpha[0][-1])
        for i in range(len(alpha[0])):
            path.append(idx)
            idx = alpha[1][-(1 + i)][idx]
        path.append(idx)

        return path[::-1]


    def total_loss(self):
        pass

    # code adapted from K.Stratos' code basis
    def score_sentence(self, score_vecs, tags, trans_matrix):
        assert(len(score_vecs)==len(tags))
        tags.insert(0, START_TAG) # add start
        total = dynet.scalarInput(.0)
        for i, obs in enumerate(score_vecs):
            # transition to next from i and emission
            next_tag = tags[i + 1]
            total += dynet.pick(trans_mat[next_tag],tags[i]) + dynet.pick(obs,next_tag)
        total += dynet.pick(trans_mat[END_TAG],tags[-1])
        return total

    # code based on https://github.com/rguthrie3/BiLSTM-CRF
    def viterbi(self, observations, unk_tag=None):
        backpointers = []
        init_vvars   = [-1e10] * self.num_tags
        init_vvars[START_TAG] = 0 # <Start> has all the probability
        for_expr     = dynet.inputVector(init_vvars)
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
                vvars_t.append(dynet.pick(next_tag_expr, best_tag_id))
            for_expr = dynet.concatenate(vvars_t) + obs
            backpointers.append(bptrs_t)
        # Perform final transition to terminal
        terminal_expr = for_expr + trans_exprs[END_TAG]
        terminal_arr  = terminal_expr.npvalue()
        best_tag_id   = np.argmax(terminal_arr)
        path_score    = dynet.pick(terminal_expr, best_tag_id)
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

    def forward(self, observations, num_tags, trans_matrix):
        # calculate forward pass
        def log_sum_exp(scores):
            npval = scores.npvalue()
            argmax_score = np.argmax(npval)
            max_score_expr = dynet.pick(scores, argmax_score)
            max_score_expr_broadcast = dynet.concatenate([max_score_expr] * num_tags)
            return max_score_expr + dynet.logsumexp_dim((scores - max_score_expr_broadcast),0)

        init_alphas = [-1e10] * num_tags
        init_alphas[START_TAG] = 0
        for_expr = dynet.inputVector(init_alphas)
        for obs in observations:
            alphas_t = []
            for next_tag in range(num_tags):
                obs_broadcast = dynet.concatenate([dynet.pick(obs, next_tag)] * num_tags)
                next_tag_expr = for_expr + trans_mat[next_tag] + obs_broadcast
                alphas_t.append(log_sum_exp(next_tag_expr))
            for_expr = dynet.concatenate(alphas_t)
        terminal_expr = for_expr + trans_mat[END_TAG]
        alpha = log_sum_exp(terminal_expr)
        return alpha































