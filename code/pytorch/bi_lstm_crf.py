
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F

from torchcrf import CRF


class PosTagger(nn.Module):

    def __init__(self, edim, hdim, voc_sz, tag_sz, emb, padix, batch_sz, dr, lstm_layers, crf):
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

        if crf:
            self.crf        = CRF(tag_sz, batch_first=True)

        # Hidden dimension
        self.hid  = self.init_hidden()

    def init_hidden(self):
        return (torch.randn(2 * self.lstm_layers, self.batch_sz, self.hdim),
                torch.randn(2 * self.lstm_layers, self.batch_sz, self.hdim))

    def forward(self, X, X_lens, mask):

        # Get the emission scores from the BiLSTM
        emissions = self._get_emission_scores(X, X_lens)

        if hasattr(self, "crf"):
            # Return best sequence
            return self.crf.decode(emissions, mask=mask)
        else:
            log_probs = F.log_softmax(emissions, dim=2)
            Y_hat = torch.argmax(log_probs, dim=2)
            return [Y_hat[i][:X_lens[i]] for i in range(len(X_lens))]


    def loss(self, X, X_lens, Y, mask=None):
        """
        Compute and return the negative log likelihood for y given X
        """

        # Get the emission scores of X
        emit_scores = self._get_emission_scores(X, X_lens)

        if hasattr(self, "crf"):
            # Get log likelihood from CRF
            log_likelihood = self.crf(emit_scores, Y, mask=mask, reduction="mean")
            loss = -log_likelihood

        else:
            # Run CrossEntropyLoss
            Y = Y.view(-1)
            Y_hat = emit_scores.view(-1, self.tag_sz)
            loss = nn.CrossEntropyLoss(ignore_index=self.padix)
            loss = loss(Y_hat, Y)

        # Return negative log likelihood
        return loss

    def _get_emission_scores(self, X, X_lens):
        """
        Runs unembedded sequences through the embedding, lstm and linear layer
        of the model and returns the emission scores of X

        Inputs: X, X_lens
            X of shape `(batch_sz, seq_len)`: input tensor
            X_lens: int list with actual lenghts of X

        Outputs: emissions
            emissions of shape `(batch_sz, seq_len, tag_sz)`: emission scores of X
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

        return X.view(batch_sz, seq_len, self.tag_sz)
