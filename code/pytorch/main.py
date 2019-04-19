import sys
import timeit

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from pytorch.models.bi_lstm import PosTagger as BiLstm
from pytorch.models.bi_lstm_crf import PosTagger as CrfBiLstm
from pytorch.reader import read_bio, read_conllu, batchify
from pytorch.school import train_epochs, train_patience, evaluate, generate_results


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
    torch.manual_seed(config["seed"])

    root, task, lang = config["data_root"], config["task"], config["language"]
    path = f"{root}{task}/{lang}"

    read, end = (read_conllu, ".conllu") if task == "pos" else (read_bio, ".bio")

    X_train, y_train = read(path + "/training"   + end)
    X_test,  y_test  = read(path + "/testing"    + end)
    X_val,   y_val   = read(path + "/validation" + end)

    embedding        = config["embedding"]

    ix2word, word2ix = create_emb_mapping(embedding)
    padix            = word2ix["<PAD>"]

    ix2tag,  tag2ix  = create_tag_mapping(y_train, padix)
    tag_sz           = len(ix2tag)

    lr              = config["learning_rate"]
    crf             = config["crf"]
    epochs          = config["epochs"]
    dropout         = config["dropout"]
    batch_sz        = config["batch_size"]
    patience        = config["patience"]
    optimizer       = config["optimizer"]
    hidden_dim      = config["hidden_size"]

    PosTagger = CrfBiLstm if crf else BiLstm

    model = PosTagger(
        hdim        = hidden_dim,
        voc_size    = embedding.shape[0],
        edim        = embedding.shape[1],
        tag_size    = tag_sz,
        embedding   = embedding,
        padix       = padix,
        batch_sz    = batch_sz,
        dropout     = dropout,
    )

    X_train, X_test, X_val = batchify(X_train, X_test, X_val, batch_sz=batch_sz, ixs=word2ix)
    y_train, y_test, y_val = batchify(y_train, y_test, y_val, batch_sz=batch_sz, ixs=tag2ix)

    if optimizer == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=lr)
    elif optimizer == "adam":
        optimizer = optim.Adam(model.parameters(), lr=lr)

    if patience:
        epochs, train_time = time(
            train_patience, model, X_train, y_train, optimizer, patience, X_val, y_val, epochs)
    else:
        epochs, train_time = time(
            train_epochs, model, X_train, y_train, optimizer, epochs)

    evaluate(model, X_test, y_test)
    (total, errors, eva), eva_time = time(
        generate_results, model, X_test, y_test, batch_sz, tag_sz)

    eva = {
        ix2tag[i]: {
            ix2tag[j]: eva[i][j].item() for j in range(tag_sz) if not j == padix
        } for i in range(tag_sz) if not i == padix
    }

    return {
        "total_values"      : total,
        "total_errors"      : errors,
        "total_oov"         : 0,
        "total_oov_errors"  : 0,
        "training_time"     : train_time,
        "evaluation_time"   : eva_time,
        "epochs_run"        : epochs,
        "evaluation_matrix" : eva
    }
