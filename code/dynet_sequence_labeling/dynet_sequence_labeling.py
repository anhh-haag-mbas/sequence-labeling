from dynet_sequence_labeling.dynet_model import DynetModel
from dynet_sequence_labeling.helper import flatten, time
from dynet_sequence_labeling.extractor import read_conllu, read_bio
from polyglot.mapping import Embedding
import sys

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
    root_path = config["data_root"]
    lang = config["language"]
    if config["task"] == "ner":
        return read_bio(root_path + f"ner/{lang}/{data_type}.bio")
    elif config["task"] == "pos":
        return read_conllu(root_path + f"pos/{lang}/{data_type}.conllu")

def create_embedding_mapping(embedding):
    vocabulary = embedding.vocabulary
    unknown_id = vocabulary.get("<UNK>")
    return lambda i: vocabulary.words[i], lambda e: vocabulary.get(e, unknown_id)

def create_mapping(elements, default = None):
    elements = list(set(elements))
    elements.sort()
    if default: elements.insert(0, default)

    ele2int = {e:i for i, e in enumerate(elements)}

    return lambda i: elements[i], lambda e: ele2int.get(e, 0)

# TODO: make cleaner
def evaluate(tagger, inputs, labels, tags, unknown):
    evaluation = {t2:{t1:0 for t1 in tags} for t2 in tags}
   
    total_words = 0
    total_oov = 0
    for s in inputs:
        for w in s:
            total_words += 1
            if w == unknown:
                total_oov += 1

    total_errors = 0
    total_oov_errors = 0

    predictions = [tagger.predict(i) for i in inputs]
    for s_preds, s_labs, s in zip(predictions, labels, inputs):
        for pred, label, word in zip(s_preds, s_labs, s):
            evaluation[label][pred] += 1
            if label != pred:
                total_errors += 1
                if word == unknown:
                    total_oov_errors += 1

    return evaluation, total_words, total_oov, total_errors, total_oov_errors

def run_experiment(config):
    validate_config(config)
    # Setup 
    train_inputs, train_labels = read_data(config, "training")
    val_inputs, val_labels     = read_data(config, "validation")
    test_inputs, test_labels   = read_data(config, "testing")
    embedding                  = config["embedding"]
    tags                       = set(flatten(train_labels))

    int2tag, tag2int           = create_mapping(tags)
    int2word, word2int         = create_embedding_mapping(embedding)

    train_inputs = [[word2int(w) for w in ws] for ws in train_inputs] 
    train_labels = [[tag2int(t) for t in ts] for ts in train_labels]

    val_inputs = [[word2int(w) for w in ws] for ws in val_inputs]
    val_labels = [[tag2int(t) for t in ts] for ts in val_labels]

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

    (evaluation, total_words, total_oov, total_errors, total_oov_errors), eva_elapsed = time(evaluate, tagger, val_inputs, val_labels, [tag2int(t) for t in tags], word2int("<UNK>"))


    evaluation = {int2tag(exp): {int2tag(act):evaluation[exp][act] for act in evaluation[exp]} for exp in evaluation}

    return {
            "total_values": total_words,
            "total_errors": total_errors,
            "total_oov": total_oov,
            "total_oov_errors": total_oov_errors,
            "training_time": elapsed,
            "evaluation_time": eva_elapsed,
            "evaluation_matrix": evaluation,
            "epochs_run": results
            }

#config = {
#        "language": "da",
#        "task": "pos",
#        "crf": False,
#        "seed": 613321,
#        "batch_size": 1,
#        "epochs": 1,
#        "patience": None,
#        "hidden_size": 100,
#        "dropout": 0.0,
#        "data_root": "../../data/"
#        }

#print(run_experiment(config))
