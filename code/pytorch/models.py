import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(1)

# START_TAG, STOP_TAG = 0, 1
START_TAG = "<START>"
STOP_TAG = "<STOP>"

def argmax(vec):
    """
    Return the argmax as a python int
    """
    _, idx = torch.max(vec, 1)
    return idx.item()

def log_sum_exp(vec):
    """
    Compute log sum exp in a numerically stable way for the forward algorithm
    """
    max_score = vec[0, argmax(vec)]
    max_score_broadcast = max_score.view(1, -1).expand(1, vec.size()[1])
    return max_score + torch.log(torch.sum(torch.exp(vec - max_score_broadcast)))

class BiPosTagger(nn.Module):

    def __init__(self, edim, hdim, vocab_size, tag_size):
        super(BiPosTagger, self).__init__()

        # Hidden dimension
        self.hdim = hdim
        self.hid  = self.init_hidden()

        # Layers
        self.embedding  = nn.Embedding(vocab_size, edim)
        self.lstm       = nn.LSTM(edim, hdim, bidirectional=True)
        self.hid2tag    = nn.Linear(hdim * 2, tag_size)

    def init_hidden(self):
        return (torch.randn(2, 1, self.hdim),
                torch.randn(2, 1, self.hdim))

    def forward(self, sentence):
        embeds           = self.embedding(sentence).view(len(sentence), 1, -1)
        out, self.hid    = self.lstm(embeds, self.hid)
        tag_space        = self.hid2tag(out.view(len(sentence), -1))
        return F.log_softmax(tag_space, dim=1)


class CrfBiPosTagger(nn.Module):

    def __init__(self, edim, hdim, vocab_size, tag2ix):
        super(CrfBiPosTagger, self).__init__()

        # Dimensions and model variables
        self.edim       = edim
        self.hdim       = hdim
        self.tag2ix     = tag2ix
        self.vocab_size = vocab_size
        self.label_size = len(tag2ix)

        # Layers
        self.embedding  = nn.Embedding(vocab_size, edim)
        self.lstm       = nn.LSTM(edim, hdim // 2, bidirectional=True)
        self.hid2tag    = nn.Linear(hdim, self.label_size)

        # Matrix of transition parameters.
        # Entry i,j is the score of transitioning _to_ i _from_ j
        self.transitions = nn.Parameter(
            torch.randn(self.label_size, self.label_size))

        # These two statements enforce the constraint that we never transfer
        # to the start tag and we never transfer from the stop tag
        self.transitions.data[tag2ix[START_TAG], :] = -10000
        self.transitions.data[:, tag2ix[STOP_TAG]]  = -10000

        # Hidden dimension
        self.hid  = self.init_hidden()

    def init_hidden(self):
        return (torch.randn(2, 1, self.hdim // 2),
                torch.randn(2, 1, self.hdim // 2))

    def _forward_alg(self, feats):
        """
        Do the forward algorithm to compute the partition function
        """
        init_alphas = torch.full((1, self.label_size), -10000.)

        # START_TAG has all of the score
        init_alphas[0][self.tag2ix[START_TAG]] = 0.

        # Wrap in a variable so that we will get automatic backprop
        forward_var = init_alphas

        # Iterate through the sentence
        for feat in feats:
            alphas_t = []   # The forward tensors at this timestep

            for next_tag in range(self.label_size):
                # broadcast the emission score: it is the same regardless of
                # the previous tag
                emit_score = feat[next_tag].view(
                    1, -1).expand(1, self.label_size)

                # the ith entry of trans_score is the score of transitioning
                # to next_tag from i
                trans_score = self.transitions[next_tag].view(1, -1)

                # The ith entry of next_tag_var is the value for the edge
                # (i -> next_tag) before we do log-sum-exp
                next_tag_var = forward_var + trans_score + emit_score

                # The forward variable for this tag is log-sum-exp of all
                # the scores
                alphas_t.append(log_sum_exp(next_tag_var).view(1))

            forward_var = torch.cat(alphas_t).view(1, -1)

        terminal_var = forward_var + self.transitions[self.tag2ix[STOP_TAG]]
        alpha = log_sum_exp(terminal_var)
        return alpha

    def _score_sentence(self, feats, tags):
        # Gives the score of a provided tag sequence
        score = torch.zeros(1)
        tags = torch.cat([torch.tensor([self.tag2ix[START_TAG]], dtype=torch.long), tags])

        for i, feat in enumerate(feats):
            score += self.transitions[tags[i + 1], tags[i]] + feat[tags[i + 1]]
        score += self.transitions[self.tag2ix[STOP_TAG], tags[-1]]
        return score

    def _viterbi_decode(self, feats):
        backpointers = []

        # Initialize the viterbi variables in log space
        init_vvars = torch.full((1, self.label_size), -10000.)
        init_vvars[0][self.tag2ix[START_TAG]] = 0

        # forward_var at step i holds the viterbi variables for step i-1
        forward_var = init_vvars
        for feat in feats:
            bptrs_t = []        # holds the backpointers for this step
            viterbivars_t = []  # holds the viterbi variables for this step

            # next_tag_var[i] holds the viterbi variable for tag i at the
            # previous step, plus the score of transitioning from tag i to next_tag
            for next_tag in range(self.label_size):
                next_tag_var = forward_var + self.transitions[next_tag]
                best_tag_id = argmax(next_tag_var)
                bptrs_t.append(best_tag_id)
                viterbivars_t.append(next_tag_var[0][best_tag_id].view(1))

            # Now add in the emission scores, and assign forward_car to the set
            # of viterbi variables we just computed
            forward_var = (torch.cat(viterbivars_t) + feat).view(1, -1)
            backpointers.append(bptrs_t)

        # Transition to STOP_TAG
        terminal_var = forward_var + self.transitions[self.tag2ix[STOP_TAG]]
        best_tag_id = argmax(terminal_var)
        path_score = terminal_var[0][best_tag_id]

        # Follow the back pointers to decode the best path
        best_path = [best_tag_id]
        for bptrs_t in reversed(backpointers):
            best_tag_id = bptrs_t[best_tag_id]
            best_path.append(best_tag_id)

        # Pop off the start tag (we don't want to return that to the caller)
        start = best_path.pop()

        # Sanity check
        assert start == self.tag2ix[START_TAG]

        best_path.reverse()
        return path_score, best_path

    def _get_lstm_features(self, sentence):
        self.hid         = self.init_hidden()
        embeds           = self.embedding(sentence).view(len(sentence), 1, -1)
        out, self.hid    = self.lstm(embeds, self.hid)
        return self.hid2tag(out.view(len(sentence), self.hdim))

    def neg_log_likelihood(self, sentence, tags):
        feats = self._get_lstm_features(sentence)
        forward_score = self._forward_alg(feats)
        gold_score = self._score_sentence(feats, tags)
        return forward_score - gold_score

    def forward(self, sentence):
        # Get the emission scores from the BiLSTM
        lstm_feats = self._get_lstm_features(sentence)

        # Find the best path, given the features
        score, tag_seq = self._viterbi_decode(lstm_feats)
        return score, tag_seq
