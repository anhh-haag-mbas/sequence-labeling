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

def _log_sum_exp(vec):
    """
    Compute log sum exp in a numerically stable way for the forward algorithm
    """
    max_score = vec[0, argmax(vec)]
    max_score_broadcast = max_score.view(1, -1).expand(1, vec.size()[1])
    return max_score + torch.log(torch.sum(torch.exp(vec - max_score_broadcast)))

def log_sum_exp(vec, m_size):
    """
    calculate log of exp sum
    args:
        vec (batch_size, vanishing_dim, hidden_dim) : input tensor
        m_size : hidden_dim
    return:
        batch_size, hidden_dim
    """
    _, idx = torch.max(vec, 1)  # B * 1 * M
    max_score = torch.gather(vec, 1, idx.view(-1, 1, m_size)).view(-1, 1, m_size)  # B * M
    sum_exp = torch.sum(torch.exp(vec - max_score.expand_as(vec)), 1)
    return max_score.view(-1, m_size) + torch.log(sum_exp).view(-1, m_size)

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
