import sys
from timeit import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import reader
from models.bi_lstm_crf import CrfBiPosTagger
from models.crf import PosTagger

torch.manual_seed(1)

PAD, UNK = 0, 1
START, STOP = 0, 1

def to_ixs(seq, ixs):
    """
    Convert words or tags (or any items in seq) to integers

    :param seq: list of words or tags
    :param ixs: dict mapping strings to integers
    :return:    tensor of dtype torch.long
    """
    wixs = [ixs.get(w, UNK) for w in seq]
    return torch.tensor(wixs, dtype=torch.long)


def train(model, X, Y, optimizer, batch_sz=32, epochs=5):
    """
    Trains a model on given data set using the provided loss function and
    optimizer and prints progress and total loss after each epoch.

    :param model:           instance of nn.Module
    :param training_data:   list of tuples, where the first entry is a sequence
                            of words and the second entry is a sequence of tags
    :param loss_func:       loss function to use, ie. nn.NLLLoss
    :param optimizer:       optimizer to use, ie. optim.SGD
    :param epochs:          number of epochs to train (defaults to 5)

    :return:                the model after training
    """
    if len(X) % batch_sz != 0:
        print(f"len(X) == {len(X)}", end="", flush=True)
        X = X[:-(len(X) % batch_sz)]
        print(f"--> len(X) == {len(X)}")

    for epoch in range(epochs):

        print(f"Training: Epoch {epoch+1}")
        print("Working... ", end="", flush=True)

        total_loss = 0
        last_batch = 0
        for batch in range(batch_sz, len(X), batch_sz):
            batch_x, x_lens, mask = prep_batch(X[last_batch:batch], word2ix)
            batch_y, _, _         = prep_batch(Y[last_batch:batch], tag2ix)
            last_batch += batch_sz

            # Print progress, because be nice
            if batch % 10 == 0:
                print("-", end="", flush=True)

            # Reset model
            model.zero_grad()

            # Make predictions and calculate loss
            loss = model.nll(batch_x, x_lens, batch_y, mask)

            # Backpropagate and train
            loss.backward()
            optimizer.step()

            # Accumulate total loss
            total_loss += loss.item()

        print(f"Epoch loss: {total_loss}")

    return model

def evaluate(model, X, Y, batch_sz=32):
    """
    Calculate, print and return the accuracy of the model in making predictions
    for the sentences in data.

    :param model:   instance of nn.Module
    :param data:    list of tuples, where the first entry is a sequence
                    of words and the second entry is a sequence of tags

    :return:        accuracy of the model
    """
    last_batch, total, correct = 0, 0, 0

    for batch in range(batch_sz, len(X), batch_sz):
        batch_x, x_lens, mask = prep_batch(X[last_batch:batch], word2ix)
        batch_y, _, _         = prep_batch(Y[last_batch:batch], tag2ix)

        pred_tags = model(batch_x, x_lens, mask)

        for i in range(batch_sz):
            pred_tag_seq = pred_tags[i]

            total += len(pred_tag_seq)

            tens = torch.ones(x_lens[0])
            tens[:len(pred_tag_seq)] = torch.tensor(pred_tag_seq)

            mask_cor = ((batch_y[i] - tens.long()) == 0).byte()
            correct += mask_cor.sum().item()

        last_batch = batch

    acc = (correct / total * 100)
    print(f"{correct} correct of {total} total. Accuracy: {acc}%")
    print()
    return acc


def prep_batch(X, ixs=None, pad_token=0):
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

    max_xlen = max(X_lens)
    batch_sz = len(X)
    padded_X = torch.ones((batch_sz, max_xlen), dtype=torch.long) * pad_token

    for i, x_len in enumerate(X_lens):
        xi = X[i] if not ixs else to_ixs(X[i], ixs)
        padded_X[i, 0:x_len] = xi[:x_len]

    idx = torch.argsort(torch.tensor(X_lens, dtype=torch.long), descending=True)
    padded_X, X_lens = padded_X[idx,:], sorted(X_lens, reverse=True)
    return padded_X, X_lens, (padded_X > PAD).byte()

##### Import data
DATA_ROOT = "../../data/"
train_filepath = DATA_ROOT + "UD_Danish-DDT/da_ddt-ud-train.conllu"
test_filepath  = DATA_ROOT + "UD_Danish-DDT/da_ddt-ud-dev.conllu"

X_train, y_train = reader.import_data(train_filepath)
X_test,  y_test  = reader.import_data(test_filepath)

##### Define word2ix, tag2ix and ix2tag
word2ix, tag2ix = { '<PAD>': PAD, '<UNK>': UNK }, { '<START>': START, '<STOP>': STOP }
for sent, tags in zip(X_train, y_train):
    for word in sent:
        word2ix[word] = word2ix.get(word, len(word2ix))

    for tag in tags:
        tag2ix[tag] = tag2ix.get(tag, len(tag2ix))

ix2tag = { ix: tag for tag, ix in tag2ix.items() }


##### Set configuration
EMBEDDING_DIM = 64
HIDDEN_DIM    = 100
VOCAB_SIZE    = len(word2ix)
LABEL_SIZE    = len(tag2ix)
BATCH_SIZE    = 8

##### Create the model
model = PosTagger(EMBEDDING_DIM, HIDDEN_DIM, VOCAB_SIZE, tag2ix, padix=PAD, batch_sz=BATCH_SIZE)

##### Either load the model state (if it exists) or train from scratch
try:
    model.load_state_dict(torch.load("./saved_models/crf.pt"))
    model.eval()
    print("Model loaded")
except FileNotFoundError:
    optimizer = optim.SGD(model.parameters(), lr=0.1)
    loss_func = nn.NLLLoss()
    start_time = time.clock()
    train(model, X_train, y_train, optimizer, batch_sz=BATCH_SIZE, epochs=3)
    print(f"Time: {time.clock() - start_time}")
    torch.save(model.state_dict(), "./saved_models/crf.pt")

##### Evaluate the model
print("Bidirectional PosTagger")
evaluate(model, X_test, y_test, batch_sz=BATCH_SIZE)
