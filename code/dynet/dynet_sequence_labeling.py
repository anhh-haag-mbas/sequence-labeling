from dynet_model import DynetModel
from helper import flatten, time
from extractor import read_conllu, read_bio
from polyglot.mapping import Embedding
from collections import defaultdict
import sys
import numpy as np

def validate_config(config):
    config_values = []
    if "task" not in config:
        config_values.append("task")
    if "language" not in config:
        config_values.append("language")
    if "crf" not in config:
        config_values.append("crf")
    if "epochs" not in config:
        config_values.append("epochs")
    if "patience" not in config:
        config_values.append("patience")
    if "seed" not in config:
        config_values.append("seed")
    if "batch_size" not in config:
        config_values.append("batch_size")
    if "hidden_size" not in config:
        config_values.append("hidden_size")

    if len(config_values) > 0:
        raise ValueError(f"No value for {config_values} in config")

def read_data(config, data_type):
    root_path = "../../data/"
    lang = config["language"]
    if config["task"] == "ner":
        return read_bio(root_path + f"ner/{lang}/{data_type}.bio")
    elif config["task"] == "pos":
        return read_conllu(root_path + f"pos/{lang}/{data_type}.conllu")

def configure_embedding(config):
    root_path = "../../data/embeddings/"
    lang = config["language"]
    return Embedding.load(root_path + f"polyglot/{lang}.tar.bz2")

def create_embedding_mapping(embedding):
    vocabulary = embedding.vocabulary
    unknown_id = vocabulary.get("<UNK>")
    return lambda i: vocabulary.words[i], lambda e: vocabulary.get(e, unknown_id)

def create_mapping(elements, default = None, embedding = None):
    elements = list(set(elements))
    elements.sort()
    if default: elements.insert(0, default)

    ele2int = {e:i for i, e in enumerate(elements)}

    return lambda i: elements[i], lambda e: ele2int.get(e, 0)

def evaluate(tagger, inputs, labels, tags):
    evaluation = {t2:{t1:0 for t1 in tags} for t2 in tags}
    predictions = [tagger.predict(i) for i in inputs]
    for s_preds, s_labs in zip(predictions, labels):
        for pred, label in zip(s_preds, s_labs):
            evaluation[label][pred] += 1
    return evaluation

def count_errors(evaluation):
    errors = 0
    for expected, actual in evaluation:
        if expected != actual:
            errors += evaluation[(expected, actual)]
    return errors

#def config_to_csv(config, separator = ","):
#    return f'{config["framework"]}{separator}{config["language"]}{separator}{config["crf"]}{separator}{config["task"]}{separator}{config["embedding_type"]}{separator}{config["seed"]}{separator}{config["mini_batches"]}{separator}{config["epochs"]}{separator}'
#
#def evaluation_to_csv(evaluation, separator = ","):
#    keys = list(evaluation.keys())
#    keys.sort()
#    values = []
#    for key in keys:
#        values.append(evaluation[key])
#    return f"{separator}".join(map(str, values))
#
#def print_evaluation(evaluation, int2tag):
#    keys = list(evaluation.keys())
#    keys.sort()
#    for t1, t2 in keys:
#        print(f"Expected {int2tag(t1)} - Actual {int2tag(t2)} = {evaluation[(t1,t2)]} times", file=sys.stderr)
#

def run_experiment(config):
    validate_config(config)
    # Setup 
    train_inputs, train_labels = read_data(config, "training")
    val_inputs, val_labels     = read_data(config, "validation")
    test_inputs, test_labels   = read_data(config, "testing")
    embedding                  = configure_embedding(config)

    words = set(flatten(train_inputs))
    tags = set(flatten(train_labels))

    int2tag, tag2int           = create_mapping(tags)
    int2word, word2int         = create_embedding_mapping(embedding)

    train_inputs = [[word2int(w) for w in ws] for ws in train_inputs] 
    train_labels = [[tag2int(t) for t in ts] for ts in train_labels]

    val_inputs = [[word2int(w) for w in ws] for ws in val_inputs]
    val_labels = [[tag2int(t) for t in ts] for ts in val_labels]

    separator = ","

    # Testing
    tagger = DynetModel(
                output_size = len(tags),
                hidden_size = config["hidden_size"],
                crf = config["crf"],
                embedding = embedding,
                seed = config["seed"],
                dropout_rate = config["dropout"])

    results, elapsed = time(
                        lambda: tagger.fit_auto_batch(
                                    sentences = train_inputs, 
                                    labels = train_labels, 
                                    mini_batch_size = config['batch_size'],
                                    epochs = config['epochs'], 
                                    patience = config["patience"], 
                                    validation_sentences = val_inputs, 
                                    validation_labels = val_labels) 
                        )
    print(results)

    evaluation, eva_elapsed = time(evaluate, tagger, val_inputs, val_labels, [tag2int(t) for t in tags])
    total_values = sum([len(i) for i in val_inputs])
    total_errors = count_errors(evaluation)
    return {
            "total_values": total_values,
            "total_errors": total_errors,
            "total_oov": total_oov,
            "total_oov_errors": total_oov_errors,
            "training_time": elapsed,
            "evaluation_time": eva_elapsed,
            "evaluation_matrix": evaluation
            }

config = {
        "language": "da",
        "task": "pos",
        "crf": False,
        "seed": 613321,
        "batch_size": 1,
        "epochs": 1,
        "patience": None,
        "hidden_size": 100,
        "dropout": 0.5
        }

print(run_experiment(config))
