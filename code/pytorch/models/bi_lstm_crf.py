
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
from torchcrf import CRF

torch.manual_seed(1)

# START_TAG, STOP_TAG = 0, 1
START_TAG = "<START>"
STOP_TAG = "<STOP>"

class PosTagger(nn.Module):

    def __init__(self, edim, hdim, vocab_size, tag2ix, padix=0, batch_sz=1):
        super(PosTagger, self).__init__()

        self.gpu = False

        # Dimensions and model variables
        self.edim       = edim
        self.hdim       = hdim
        self.tag2ix     = tag2ix
        self.tag_size   = vocab_size
        self.tag_size   = len(tag2ix)
        self.batch_sz   = batch_sz

        # Layers
        self.embedding  = nn.Embedding(vocab_size, edim, padding_idx=padix)
        self.lstm       = nn.LSTM(edim, hdim // 2, bidirectional=True, batch_first=True)
        self.hid2tag    = nn.Linear(hdim, self.tag_size)
        self.crf        = CRF(self.tag_size, batch_first=True)

        # Hidden dimension
        self.hid  = self.init_hidden()

    def init_hidden(self):
        return (torch.randn(2, self.batch_sz, self.hdim // 2),
                torch.randn(2, self.batch_sz, self.hdim // 2))

    def nll(self, X, X_lens, Y, mask=None):
        """
        Compute and return the negative log likelihood for y given X
        """

        # Get the emission scores of X
        X_emit = self._get_emission_scores(X, X_lens)

        # Get log likelihood from CRF
        log_likelihood = self.crf(X_emit, Y, mask=mask)

        # Return negative log likelihood
        return -log_likelihood

    def forward(self, X, X_lens, mask):

        # Get the emission scores from the BiLSTM
        emissions = self._get_emission_scores(X, X_lens)

        # Return best sequence
        return self.crf.decode(emissions, mask=mask)

    def _get_emission_scores(self, X, X_lens):
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

        X = torch.nn.utils.rnn.pack_padded_sequence(X, X_lens, batch_first=True)
        X, self.hid = self.lstm(X, self.hid)
        X, _ = torch.nn.utils.rnn.pad_packed_sequence(X, batch_first=True)

        X = self.hid2tag(X.contiguous().view(-1, X.size(2)))

        return X.view(batch_sz, seq_len, self.tag_size)
