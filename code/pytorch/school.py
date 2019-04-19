import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

def _train(model, X, Y, optimizer):
    model.train()

    total_loss = 0
    for batch in range(len(X)):
        batch_x, x_lens, mask = X[batch]
        batch_y, _, _         = Y[batch]

        # Print progress, because be nice
        if batch % 10 == 0:
            print("-", end="", flush=True)

        # Reset model
        model.zero_grad()

        # Make predictions and calculate loss
        loss = model.loss(batch_x, x_lens, batch_y, mask)

        # Backpropagate and train
        loss.backward()
        optimizer.step()

        # Accumulate total loss
        total_loss += loss.item()

    return total_loss


def train_patience(model, X, Y, optimizer, patience, X_val, Y_val, max_epochs):
    batch_sz = X[0][0].size(0)
    acc = evaluate(model, X_val, Y_val)
    epoch, counter = 0, 0

    while counter < patience and epoch < max_epochs:
        epoch += 1

        print(f"Counter is {counter}")

        print(f"Training: Epoch {epoch}")
        print("Working... ", end="", flush=True)
        total_loss = _train(model, X, Y, optimizer)
        print(f"Epoch loss: {total_loss}")

        curr_acc = evaluate(model, X_val, Y_val)
        if curr_acc > acc:
            acc = curr_acc
            counter = 0
        else:
            counter += 1


def train_epochs(model, X, Y, optimizer, epochs):
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

    for epoch in range(epochs):

        print(f"Training: Epoch {epoch+1}")
        print("Working... ", end="", flush=True)
        total_loss = _train(model, X, Y, optimizer)
        print(f"Epoch loss: {total_loss}")

def evaluate(model, X, Y):
    """
    Calculate, print and return the accuracy of the model in making predictions
    for the sentences in data.

    :param model:   instance of nn.Module
    :param data:    list of tuples, where the first entry is a sequence
                    of words and the second entry is a sequence of tags

    :return:        accuracy of the model
    """
    last_batch, total, correct = 0, 0, 0

    model.eval()
    for batch in range(len(X)):
        batch_x, x_lens, mask = X[batch]
        batch_y, _, _         = Y[batch]

        batch_sz, seq_len = batch_x.size()

        pred_tags = model(batch_x, x_lens, mask)

        for i in range(batch_sz):
            pred_tag_seq = pred_tags[i]

            total += x_lens[i]

            ones = torch.ones(seq_len)
            ones[:len(pred_tag_seq)] = torch.tensor(pred_tag_seq)

            mask_cor = ((batch_y[i] - ones.long()) == 0).byte()
            correct += mask_cor.sum().item()

        last_batch = batch

    acc = (correct / total * 100)
    print(f"{correct} correct of {total} total. Accuracy: {acc}%")
    print()
    return acc


def generate_results(model, X, Y, batch_sz, tag_sz):
    model.eval()
    evaluation = torch.zeros(tag_sz, tag_sz)

    for batch in range(len(X)):
        batch_x, x_lens, mask = X[batch]
        batch_y, _, _         = Y[batch]

        pred_tags = model(batch_x, x_lens, mask)

        for i in range(batch_sz):
            pred_tag_seq = pred_tags[i]

            for pred, act in zip(pred_tag_seq, batch_y[i][:x_lens[i]]):
                evaluation[pred][act] += 1

    total   = torch.sum(evaluation).item()
    correct = int(sum([evaluation[i][i] for i in range(tag_sz)]))
    acc = correct / total * 100

    print(f"{correct} correct of {total} total. Accuracy: {acc}%")
    print()

    return total, (total - correct), evaluation.int()
