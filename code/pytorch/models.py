import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(1)

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
