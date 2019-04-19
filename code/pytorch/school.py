import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

def _train(model, X, Y, optimizer):
    model.train()

    print("Working... ", end="", flush=True)
    print(f"Number of batches: {len(X)}")
    k = len(X) // 100

    total_loss = 0
    for batch in range(len(X)):
        batch_x, x_lens, mask = X[batch]
        batch_y, _, _         = Y[batch]

        # Print progress, because be nice
        if batch % k == 0:
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

    print()
    return total_loss


def train_patience(model, X, Y, optimizer, patience, X_val, Y_val, max_epochs):
    batch_sz = X[0][0].size(0)
    best_acc = evaluate(model, X_val, Y_val)
    epochs, counter = 0, 0

    while counter < patience and epochs < max_epochs:
        epochs += 1

        print(f"Patience left {patience - counter}")

        print(f"Training: Epoch {epochs}")
        total_loss = _train(model, X, Y, optimizer)
        print(f"Epoch loss: {total_loss}")

        curr_acc = evaluate(model, X_val, Y_val)
        if curr_acc > best_acc:
            best_acc = curr_acc
            counter = 0
        else:
            counter += 1

    return epochs


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


def generate_results(model, X, Y, tag_sz, unkix):
    model.eval()
    evaluation = torch.zeros(tag_sz, tag_sz)

    nb_oov        = 0
    nb_oov_errors = 0

    for batch in range(len(X)):
        batch_x, x_lens, mask = X[batch]
        batch_y, _, _         = Y[batch]
        batch_sz = batch_x.shape[0]

        Y_hat = model(batch_x, x_lens, mask)

        for i in range(batch_sz):
            Yi     = batch_y[i][:x_lens[i]]
            Xi     = batch_x[i][:x_lens[i]]
            Yi_hat = Y_hat[i]

            for word, pred, act in zip(Xi, Yi_hat, Yi):
                if word == unkix:
                    nb_oov        += 1
                    nb_oov_errors += 1 if not pred == act else 0

                evaluation[pred][act] += 1

    total   = int(torch.sum(evaluation).item())
    correct = int(sum([evaluation[i][i] for i in range(tag_sz)]))
    errors  = total - correct
    acc = correct / total * 100

    print(f"{correct} correct of {total} total. Accuracy: {acc}%")
    print()

    return total, errors, nb_oov, nb_oov_errors, evaluation.int()
