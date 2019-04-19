import sys
from timeit import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from pytorch.models.bi_lstm import PosTagger as BiLstm
from pytorch.models.bi_lstm_crf import PosTagger as CrfBiLstm
from pytorch.reader import import_bio, import_conllu, batchify
from pytorch.school import train_epochs, train_patience, evaluate


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

    if task == "pos":
        X_train, y_train = import_conllu(path + "/training.conllu")
        X_test,  y_test  = import_conllu(path + "/testing.conllu")
        X_val,   y_val   = import_conllu(path + "/validation.conllu")
    elif task == "ner":
        X_train, y_train = import_bio(path + "/training.bio")
        X_test,  y_test  = import_bio(path + "/testing.bio")
        X_val,   y_val   = import_bio(path + "/validation.bio")

    embedding        = config["embedding"]
    ix2word, word2ix = create_emb_mapping(embedding)
    ix2tag,  tag2ix  = create_tag_mapping(y_train, word2ix['<PAD>'])

    lr              = config["learning_rate"]
    crf             = config["crf"]
    epochs          = config["epochs"]
    dropout         = config["dropout"]
    patience        = config["patience"]
    optimizer       = config["optimizer"]
    batch_size      = config["batch_size"]
    hidden_dim      = config["hidden_size"]

    PosTagger = CrfBiLstm if crf else BiLstm

    model = PosTagger(
        hdim        = hidden_dim,
        voc_size    = embedding.shape[0],
        edim        = embedding.shape[1],
        tag_size    = len(tag2ix),
        embedding   = embedding,
        padix       = word2ix["<PAD>"],
        batch_sz    = batch_size,
        dropout     = dropout,
    )

    X_train, X_test, X_val = batchify(X_train, X_test, X_val, batch_sz=batch_size, ixs=word2ix)
    y_train, y_test, y_val = batchify(y_train, y_test, y_val, batch_sz=batch_size, ixs=tag2ix)

    if optimizer == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=lr)

    if patience:
        train_patience(
            model, X_train, y_train, optimizer, patience, X_val, y_val, epochs)
    else:
        train_epochs(model, X_train, y_train, optimizer, epochs)

    evaluate(model, X_test, y_test, batch_sz=batch_size)

    return {
        "total_values"      : 0,
        "total_errors"      : 0,
        "total_oov"         : 0,
        "total_oov_errors"  : 0,
        "training_time"     : 0,
        "evaluation_time"   : 0,
        "epochs_run"        : 0
    }
