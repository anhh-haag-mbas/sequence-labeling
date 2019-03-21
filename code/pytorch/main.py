import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import models
import reader

torch.manual_seed(1)


##### Import data
DATA_ROOT = "../../data/"
train_filepath = DATA_ROOT + "UD_Danish-DDT/da_ddt-ud-train.conllu"
test_filepath  = DATA_ROOT + "UD_Danish-DDT/da_ddt-ud-dev.conllu"

train_data = reader.import_data(train_filepath)
test_data  = reader.import_data(test_filepath)

##### Define word2ix, tag2ix and ix2tag
word2ix, tag2ix = { '<UNK>': 0 }, { '<START>': 0, '<END>': 1 }
for sent, tags in train_data:
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

##### Create the model
model = models.BiPosTagger(EMBEDDING_DIM, HIDDEN_DIM, VOCAB_SIZE, LABEL_SIZE)

##### Either load the model state (if it exists) or train from scratch
try:
    model.load_state_dict(torch.load("./saved_models/biLSTM.pt"))
    model.eval()
    print("Model loaded")
except FileNotFoundError:
    optimizer = optim.SGD(model.parameters(), lr=0.1)
    loss_func = nn.NLLLoss()
    train(model, train_data, loss_func, optimizer)
    torch.save(model.state_dict(), "./saved_models/biLSTM.pt")

##### Evaluate the model
print("Bidirectional PosTagger")
evaluate(model, test_data)


def to_ixs(seq, ixs):
    """
    Convert words or tags (or any items in seq) to integers

    :param seq: list of words or tags
    :param ixs: dict mapping strings to integers
    :return:    tensor of dtype torch.long
    """
    wixs = [ixs.get(w, 0) for w in seq]
    return torch.tensor(wixs, dtype=torch.long)


def train(model, training_data, loss_func, optimizer, epochs=5):
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
    remaining = len(training_data)
    for epoch in range(epochs):

        print("Training: Epoch {}".format(epoch))
        print("Working... ", end="", flush=True)

        total_loss = 0
        for sent, tags in training_data:

            # Print progress, because be nice
            remaining -= 1
            if remaining % 100 == 0:
                print("-", end="", flush=True)

            # Reset model
            model.zero_grad()
            model.hidden = model.init_hidden()

            # Make predictions and calculate loss
            logprobs = model(to_ixs(sent, word2ix))
            loss = loss_func(logprobs, to_ixs(tags, tag2ix))

            # Backpropagate and train
            loss.backward()
            optimizer.step()

            # Accumulate total loss
            total_loss += loss.item()

        print("Epoch loss: {}".format(total_loss))

    return model

def evaluate(model, data):
    """
    Calculate, print and return the accuracy of the model in making predictions
    for the sentences in data.

    :param model:   instance of nn.Module
    :param data:    list of tuples, where the first entry is a sequence
                    of words and the second entry is a sequence of tags

    :return:        accuracy of the model
    """
    total, correct = 0, 0

    for sent, tags in data:
        sentence = to_ixs(sent, word2ix)
        tag_ixs = to_ixs(tags, tag2ix)
        model.hidden = model.init_hidden()
        out = model(sentence)

        for scores, word, tag_ix in zip(out, sent, tag_ixs):
            exp_tag = ix2tag[tag_ix.item()]
            pred_tag = ix2tag[torch.argmax(scores).item()]

            total += 1
            if exp_tag == pred_tag:
                correct += 1

    acc = (correct / total * 100)
    print("{} correct of {} total. Accuracy: {}%".format(correct, total, acc))
    print()
    return acc
