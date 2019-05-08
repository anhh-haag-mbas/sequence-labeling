import sys
import timeit

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from pytorch.bi_lstm_crf import PosTagger
from pytorch.reader import read_bio, read_conllu, batchify
from pytorch.school import train_epochs, train_patience, evaluate, generate_results

torch.set_num_threads(1)

DEFAULT_LSTM_LAYERS = 1

def time(func, *args):
    """
    I totally stole this
    """
    start_time = timeit.default_timer()
    result = func(*args)
    elapsed = timeit.default_timer() - start_time
    return (result, elapsed)


def create_emb_mapping(embedding):
    vocabulary = embedding.vocabulary
    return vocabulary.id_word, vocabulary.word_id

def create_tag_mapping(tags_list, padix):
    tag2ix, ix2tag = {}, {}
    for tags in tags_list:
        for tag in tags:
            ix = tag2ix.get(tag, len(tag2ix))
            tag2ix[tag] = ix
            ix2tag[ix] = tag

    newix = len(tag2ix)
    tag2ix['<PAD>'] = padix
    tag = ix2tag[padix]

    tag2ix[tag] = newix
    ix2tag[padix] = '<PAD>'
    ix2tag[newix] = tag

    return ix2tag, tag2ix


def run_experiment(config):
    # Seed shit
    torch.manual_seed(config["seed"])

    # Define path, task and lang
    root, task, lang = config["data_root"], config["task"], config["language"]
    path = f"{root}{task}/{lang}"

    # Get an ending and a read function dependent on the task
    read, end = (read_conllu, ".conllu") if task == "pos" else (read_bio, ".bio")

    # Fetch raw data
    X_train, y_train = read(path + "/training"   + end)
    X_test,  y_test  = read(path + "/testing"    + end)
    X_val,   y_val   = read(path + "/validation" + end)

    # Load polyglot embedding
    embedding        = config["embedding"]

    # Define dictionaries and find the padding index and the tag size
    ix2word, word2ix = create_emb_mapping(embedding)
    padix            = word2ix["<PAD>"]

    ix2tag,  tag2ix  = create_tag_mapping(y_train, padix)
    tag_sz           = len(ix2tag)

    # Set configurations
    lr              = config["learning_rate"]
    crf             = config["crf"]
    epochs          = config["epochs"]
    dropout         = config["dropout"]
    batch_sz        = config["batch_size"]
    patience        = config["patience"]
    optimizer       = config["optimizer"]
    hidden_dim      = config["hidden_size"]
    lstm_layers     = config.get("lstm_layers", DEFAULT_LSTM_LAYERS)

    # Define and instantiate the POS Tagger
    model = PosTagger(
        hdim        = hidden_dim,
        voc_sz      = embedding.shape[0],
        edim        = embedding.shape[1],
        tag_sz      = tag_sz,
        emb         = embedding,
        padix       = padix,
        batch_sz    = batch_sz,
        dr          = dropout,
        lstm_layers = lstm_layers,
        crf         = crf,
    )

    # Create the opimizer
    if optimizer == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=lr)
    elif optimizer == "adam":
        optimizer = optim.Adam(model.parameters(), lr=lr)

    # Batchify data (list of (tensor: batch, list: lengths, tensor: mask) )
    X_train, X_test, X_val = batchify(X_train, X_test, X_val, batch_sz=batch_sz, ixs=word2ix)
    y_train, y_test, y_val = batchify(y_train, y_test, y_val, batch_sz=batch_sz, ixs=tag2ix)

    # Either train with patience or fixed amount of epochs (and time it)
    if patience:
        epochs, train_time = time(
            train_patience, model, X_train, y_train, optimizer, patience, X_val, y_val, epochs)
    else:
        _, train_time = time(
            train_epochs, model, X_train, y_train, optimizer, epochs)

    # Generate evaluation results and time how long it took
    (total, errors, nb_oov, nb_oov_errors, eva), eva_time = time(
        generate_results, model, X_test, y_test, tag_sz, word2ix['<UNK>'])

    # Convert evaluation matrix to map from tag names instead of tag ids
    eva = {
        ix2tag[i]: {
            ix2tag[j]: eva[i][j].item() for j in range(tag_sz) if not j == padix
        } for i in range(tag_sz) if not i == padix
    }

    import ipdb; ipdb.set_trace()

    # Return dict of results
    return {
        "total_values"      : total,
        "total_errors"      : errors,
        "total_oov"         : nb_oov,
        "total_oov_errors"  : nb_oov_errors,
        "training_time"     : train_time,
        "evaluation_time"   : eva_time,
        "epochs_run"        : epochs,
        "evaluation_matrix" : eva
    }
