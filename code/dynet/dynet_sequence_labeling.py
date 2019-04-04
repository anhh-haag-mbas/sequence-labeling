from dynet_model import DynetModel
from helper import flatten, time
from extractor import read_conllu, read_bio
from polyglot.mapping import Embedding
import sys
import numpy as np

unknown = "<UNK>"

def validate_config(config):
    config_values = []
    if "task" not in config:
        config_values.append("task")
    if "language" not in config:
        config_values.append("language")
    if "embedding_type" not in config:
        config_values.append("embedding_type")
#    elif "embedding" not in config and config["embedding_type"] != "self_trained":
#        config_values.append("embedding")
#    else "embedding_dimensions" not in config:
#        config_values.append("embedding_dimensions")
    if "seed" not in config:
        config_values.append("seed")

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
    if config["embedding_type"] == "fasttext":
        raise NotImplemented("fasttext support missing")
    elif config["embedding_type"] == "polyglot":
        embedding = Embedding.load(root_path + f"polyglot/{lang}.tar.bz2")
        return embedding
    return None

def create_embedding_mapping(embedding):
    vocabulary = embedding.vocabulary
    unknown_id = vocabulary.get("<UNK>")
    return lambda i: vocabulary.words[i], lambda e: vocabulary.get(e, unknown_id)

def create_mapping(elements, default = None, embedding = None):
    if embedding is not None: return create_embedding_mapping(embedding)

    elements = list(set(elements))
    elements.sort()
    if default: elements.insert(0, default)

    ele2int = {e:i for i, e in enumerate(elements)}

    return lambda i: elements[i], lambda e: ele2int.get(e, 0)

def evaluate(tagger, inputs, labels, tags):
    evaluation = {(t1, t2):0 for t1 in range(len(tags)) for t2 in range(len(tags))}
    predictions = [tagger.predict(i) for i in inputs]
    for s_preds, s_labs in zip(predictions, labels):
        for pred, label in zip(s_preds, s_labs):
            evaluation[(label, pred)] += 1
    return evaluation

def count_errors(evaluation):
    errors = 0
    for expected, actual in evaluation:
        if expected != actual:
            errors += evaluation[(expected, actual)]
    return errors

def config_to_csv(config, separator = ","):
    return f'{config["framework"]}{separator}{config["language"]}{separator}{config["crf"]}{separator}{config["task"]}{separator}{config["embedding_type"]}{separator}{config["seed"]}{separator}{config["mini_batches"]}{separator}{config["epochs"]}{separator}'

def evaluation_to_csv(evaluation, separator = ","):
    keys = list(evaluation.keys())
    keys.sort()
    values = []
    for key in keys:
        values.append(evaluation[key])
    return f"{separator}".join(map(str, values))

def print_evaluation(evaluation, int2tag):
    keys = list(evaluation.keys())
    keys.sort()
    for t1, t2 in keys:
        print(f"Expected {int2tag(t1)} - Actual {int2tag(t2)} = {evaluation[(t1,t2)]} times", file=sys.stderr)


def run_experiment(config):
    validate_config(config)
    # Setup 
    train_inputs, train_labels = read_data(config, "training")
    val_inputs, val_labels     = read_data(config, "validation")
    embedding                  = configure_embedding(config)

    words = set(flatten(train_inputs))
    tags = set(flatten(train_labels))

    int2tag, tag2int           = create_mapping(tags)
    int2word, word2int         = create_mapping(words, default = "<UNK>", embedding = embedding)

#    train_inputs = [apply_dropout(ws, config['dropout']) for ws in train_inputs]
    train_inputs = [[word2int(w) for w in ws] for ws in train_inputs] 
    train_labels = [[tag2int(t) for t in ts] for ts in train_labels]

    val_inputs = [[word2int(w) for w in ws] for ws in val_inputs]
    val_labels = [[tag2int(t) for t in ts] for ts in val_labels]

    separator = ","

    # Testing
    tagger = DynetModel(
                input_size = len(words) + 1,
                output_size = len(tags),
                embed_size = config["embedding_size"],
                hidden_size = config["hidden_size"],
                crf = config["crf"],
                embedding = embedding,
                seed = config["seed"],
                dropout_rate = config["dropout"])

    results, elapsed = time(
                        lambda: tagger.fit_auto_batch(
                                    sentences = train_inputs, 
                                    labels = train_labels, 
                                    mini_batch_size = config['mini_batches'],
                                    epochs = config['epochs'], 
                                    patience = config["patience"], 
                                    validation_sentences = val_inputs, 
                                    validation_labels = val_labels) 
                        )
    print(results)

    evaluation, eva_elapsed = time(evaluate, tagger, val_inputs, val_labels, tags)
    total_values = sum([len(i) for i in val_inputs])
    total_errors = count_errors(evaluation)
    accuracy = 1 - (total_errors / total_values)
    print(f"training {elapsed} - test {eva_elapsed}, tokens {total_values} - errors {total_errors}")
    return config_to_csv(config) + f"{elapsed}{separator}{total_errors}{separator}{accuracy}{separator}" + evaluation_to_csv(evaluation)

#    for expected, actual in evaluation.keys():
#        key = (expected, actual)
#        if evaluation[key] != 0:
#            print(f"Expected {int2tag[expected]} - actual {int2tag[actual]} = {evaluation[key]} times")
#    print(f"Training time {elapsed}")
#    print(f"Training output {results}")

languages = ["da"]#["ar", "de", "el", "en", "fr", "hu", "ja", "ko", "ru", "sk", "tr", "zh"]
tasks = ["pos"]
seeds = [659054003] #, 319650727, 614680916, 686244532, 3846303]

for lang in languages:
    for seed in seeds:
        for task in tasks:
            config = {
                    "crf": False,
                    "embedding_size": 86,
                    "hidden_size": 100,
                    "seed": seed, 
                    "dropout": 0.50,
                    "task": task, 
                    "language": lang, 
                    "embedding_type": "polyglot",
                    #"embedding_type": "self_trained",
                    "framework": "dynet",
                    "mini_batches": 16,
                    "epochs": 20,
                    "patience": 3
                    }

            print(run_experiment(config))
