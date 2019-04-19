
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F

from torchcrf import CRF

START_TAG = "<START>"
STOP_TAG = "<STOP>"

class PosTagger(nn.Module):

    def __init__(self, edim, hdim, voc_size, tag_size, embedding, padix, batch_sz, dropout):
        super(PosTagger, self).__init__()

        self.gpu = False

        # Dimensions and model variables
        self.edim       = edim
        self.hdim       = hdim
        self.padix      = padix
        self.voc_size   = voc_size
        self.tag_size   = tag_size
        self.batch_sz   = batch_sz

        # Layers
        self.embedding  = nn.Embedding(voc_size, edim, padding_idx=padix)
        if embedding:
            self.embedding.load_state_dict({'weight': torch.tensor(embedding.vectors)})

        self.dropout    = nn.Dropout(dropout)
        self.lstm       = nn.LSTM(edim, hdim // 2, bidirectional=True, batch_first=True)
        self.hid2tag    = nn.Linear(hdim, tag_size)

        # Hidden dimension
        self.hid  = self.init_hidden()

    def init_hidden(self):
        return (torch.randn(2, self.batch_sz, self.hdim // 2),
                torch.randn(2, self.batch_sz, self.hdim // 2))

    def forward(self, X, X_lens, mask):
        return torch.argmax(self._forward(X, X_lens), dim=2)

    def loss(self, X, X_lens, Y, mask=None):
        """
        Compute and return the negative log likelihood for y given X
        """
        log_probs = self._forward(X, X_lens)

        Y = Y.view(-1)
        Y_hat = log_probs.view(-1, self.tag_size)
        loss = nn.NLLLoss(ignore_index=self.padix)

        return loss(Y_hat, Y)


    def _forward(self, X, X_lens):
        """
        Runs unembedded sequences through the embedding, lstm and linear layer
        of the model and returns the emission scores of X

        Inputs: X, X_lens
            X of shape `(batch_sz, seq_len)`: input tensor
            X_lens: int list with actual lenghts of X

        Outputs: emissions
            emissions of shape `(batch_sz, seq_len, tag_size)`: emission scores of X
        """
        self.hid = self.init_hidden()

        batch_sz, seq_len = X.size()
        X = self.embedding(X)
        X = self.dropout(X)

        X = torch.nn.utils.rnn.pack_padded_sequence(X, X_lens, batch_first=True)
        X, self.hid = self.lstm(X, self.hid)
        X, _ = torch.nn.utils.rnn.pad_packed_sequence(X, batch_first=True)

        X = self.dropout(X)

        X = self.hid2tag(X.contiguous().view(-1, X.size(2)))

        X = F.log_softmax(X, dim=1)

        return X.view(batch_sz, seq_len, self.tag_size)
