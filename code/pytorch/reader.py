import torch
import numpy as np


PAD_IX,   UNK_IX      = 0, 1
PAD_WORD, UNK_WORD    = "<PAD>", "<UNK>"

START_IX,  STOP_IX    = 0, 1
START_TAG, STOP_TAG   = "<START>", "<STOP>"


FORM, UPOS = 1, 3
def read_conllu(path):
    X, y = [], []
    words, tags = [], []
    with open(path, "r", encoding="utf-8") as f:
        for line in f.readlines():
            line.strip()

            if line.startswith("#"):
                continue

            if line.isspace():
                X.append(words)
                y.append(tags)
                words, tags = [], []
                continue

            line = line.split("\t")
            if len(line) == 2:
                words.append(line[0])
                tags.append(line[1])
            else:
                words.append(line[FORM])
                tags.append(line[UPOS])

    return X, y


def read_bio(path):
    X, y = [], []
    words, tags = [], []
    with open(path, "r", encoding="utf-8") as f:
        for line in f.readlines():
            line.strip()

            if line.startswith("#"):
                continue

            if line.isspace():
                X.append(words)
                y.append(tags)
                words, tags = [], []

            else:
                w, t = line.split("\t")
                words.append(w)
                tags.append(t)

    return X, y



def create_mapping(words, tags):
    tag2ix  = { START_TAG: START_IX, STOP_TAG: STOP_IX }
    word2ix = { PAD_WORD: PAD_IX, UNK_WORD: UNK_IX}

    for sent, tags in zip(words, tags):
        for word in sent:
            word2ix[word] = word2ix.get(word, len(word2ix))

        for tag in tags:
            tag2ix[tag] = tag2ix.get(tag, len(tag2ix))


    return word2ix, tag2ix


def to_ixs(seq, ixs):
    """
    Convert words or tags (or any items in seq) to integers

    :param seq: list of words or tags
    :param ixs: dict mapping strings to integers
    :return:    tensor of dtype torch.long
    """
    UNK = ixs.get('<UNK>', -1)
    wixs = [ixs.get(w, UNK) for w in seq]
    return torch.tensor(wixs, dtype=torch.long)


def batchify(*args, batch_sz=1, ixs=None):
    """
    Takes multiple datasets and batches them in batch sizes of `batch_sz`
    """
    for arg in args:
        yield _batchify(arg, batch_sz, ixs)


def _batchify(data, batch_sz, ixs):
    """
    Batch a dataset in batches of size `batch_sz` and convert to integer
    representation according to `ixs`
    """
    # rem = (len(data) % batch_sz)
    # data = data if rem == 0 else data[:-rem]

    out = []
    i = 0
    for j in range(batch_sz, len(data) + 1, batch_sz):
        out.append(prep_batch(data[i:j], ixs))
        i = j

    return out


def prep_batch(X, ixs):
    """
    Take a batch of data X and returns same sized padded sequences of integer
    representation of the X. If ixs is not provided, X is assumed to already
    contain the integer representation

    output:
        padded_X:   (batch_sz, seq_len)
        X_lens:     list of original lenghts of X's
        mask:       (batch_sz, seq_len)
    """
    X_lens = [len(x) for x in X]
    pad_token = ixs["<PAD>"]

    max_xlen = max(X_lens)
    batch_sz = len(X)
    padded_X = torch.ones((batch_sz, max_xlen), dtype=torch.long) * pad_token

    for i, x_len in enumerate(X_lens):
        xi = X[i] if not ixs else to_ixs(X[i], ixs)
        padded_X[i, 0:x_len] = xi[:x_len]

    idx = torch.argsort(torch.tensor(X_lens, dtype=torch.long), descending=True)

    padded_X, X_lens = padded_X[idx,:], sorted(X_lens, reverse=True)
    return padded_X, X_lens, (padded_X != pad_token).byte()
