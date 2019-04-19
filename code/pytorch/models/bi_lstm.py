
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F

from torchcrf import CRF


class PosTagger(nn.Module):

    def __init__(self, edim, hdim, voc_sz, tag_sz, emb, padix, batch_sz, dr, lstm_layers):
        super(PosTagger, self).__init__()

        self.gpu = False

        # Dimensions and model variables
        self.edim        = edim
        self.hdim        = hdim
        self.padix       = padix
        self.voc_sz      = voc_sz
        self.tag_sz      = tag_sz
        self.batch_sz    = batch_sz
        self.lstm_layers = lstm_layers

        # Layers
        self.embedding  = nn.Embedding(voc_sz, edim, padding_idx=padix)
        if emb:
            self.embedding.load_state_dict({'weight': torch.tensor(emb.vectors)})

        self.dropout    = nn.Dropout(p=dr)
        self.lstm       = nn.LSTM(
            edim,
            hdim,
            num_layers=lstm_layers,
            bidirectional=True,
            batch_first=True
        )
        self.hid2tag    = nn.Linear(hdim * 2, tag_sz)

        # Hidden dimension
        self.hid  = self.init_hidden()

    def init_hidden(self):
        return (torch.randn(2 * self.lstm_layers, self.batch_sz, self.hdim),
                torch.randn(2 * self.lstm_layers, self.batch_sz, self.hdim))

    def forward(self, X, X_lens, mask):
        out = self._forward(X, X_lens)
        return torch.argmax(out, dim=2)

    def loss(self, X, X_lens, Y, mask=None):
        """
        Compute and return the negative log likelihood for y given X
        """
        log_probs = self._forward(X, X_lens)

        Y = Y.view(-1)
        Y_hat = log_probs.view(-1, self.tag_sz)
        loss = nn.NLLLoss(ignore_index=self.padix)

        return loss(Y_hat, Y)


    def _forward(self, X, X_lens):
        """
        Runs unembedded sequences through the embedding, lstm and linear layer
        of the model and returns the emission scores of X

        Inputs: X, X_lens
            X of shape `(batch_sz, seq_len)`: input tensor
            X_lens: int list with actual lenghts of X

        Outputs: log probabilities
            log probs of shape `(batch_sz, seq_len, tag_sz)`: log probs of X
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

        return X.view(batch_sz, seq_len, self.tag_sz)
