import sys
import os
import requests

URL = os.environ["URL"]
assert URL is not None

if sys.argv[1] == "pytorch":
    import pytorch.main as pyt
if sys.argv[1] == "dynet":
    import dynet_sequence_labeling.dynet_sequence_labeling as dysl
if sys.argv[1] == "tensorflow":
    from tensorflow_sequence_labelling.model import TensorFlowSequenceLabelling as tf

from itertools import product
from polyglot.mapping import Embedding

def validate_results(results):
    keys = [
        "total_values",
        "total_errors",
        "total_oov",
        "total_oov_errors",
        "training_time",
        "evaluation_time",
        "epochs_run"
    ]

    for key in keys:
        if key not in results.keys():
            with open("errors", "a", encoding = "utf-8") as of:
                of.write(f"Missing value {key} from experiment results")
            raise ValueError(f"Missing value {key} from experiment results")

def config_to_str(config, separator):
    keys = [
        "framework",
        "language",
        "task",
        "crf",
        "seed",
        "batch_size",
        "epochs",
        "patience"
    ]

    values = map(lambda k: str(config[k]), keys)
    return separator.join(values)

def evaluation_to_str(evaluation, separator):
    keys = list(evaluation.keys())
    keys.sort()
    values = []
    for expected, actual in product(keys, keys):
        values.append(str(evaluation[expected][actual]))

    return separator.join(values)

def results_to_str(results, separator):
    keys = [
        "total_values",
        "total_errors",
        "total_oov",
        "total_oov_errors",
        "training_time",
        "evaluation_time",
        "epochs_run"
    ]

    values = map(lambda k: str(results[k]), keys)
    return separator.join(values) + separator + evaluation_to_str(results["evaluation_matrix"], separator)

def run_experiment(config):
    framework = config["framework"]
    if framework == "dynet":
        return dysl.run_experiment(config)
    if framework == "pytorch":
        return run_experiment_pytorch(config)
    if framework == "tensorflow":
        return run_experiment_tensorflow(config)

def run_experiment_pytorch(config):
    return pyt.run_experiment(config)

def run_experiment_tensorflow(config):
    return tf(config).run()

def experiment_to_str(config, results):
    separator = ","
    return config_to_str(config,separator)+separator+results_to_str(results, separator)+"\n"

def load_embeddings(data_root, languages):
    return {l:Embedding.load(data_root + (f"embeddings/{l}.tar.bz2")) for l in languages}

data_root   = "../data/"
languages   = ["da", "no", "ru", "hi", "ur", "ja", "ar"]
embeddings = load_embeddings(data_root, languages)

config = {
        "framework"     : sys.argv[1],
        "language"      : sys.argv[2],
        "task"          : sys.argv[3],
        "crf"           : bool(sys.argv[4]),
        "seed"          : int(sys.argv[5]),
        "batch_size"    : int(sys.argv[6]),
        "epochs"        : int(sys.argv[7]),
        "patience"      : None if sys.argv[8] == "None" else int(sys.argv[8]),
        "hidden_size"   : 100,
        "dropout"       : 0.5,
        "data_root"     : data_root,
        "embedding"     : embeddings[sys.argv[2]],
        "optimizer"     : "sgd",
        "learning_rate" : 0.1,
        }
results = run_experiment(config)
validate_results(results)
with open("out.csv", "a", encoding = "utf-8") as of:
    of.write(experiment_to_str(config, results))
requests.post(URL + "/result", json=experiment_to_str(config, results))
