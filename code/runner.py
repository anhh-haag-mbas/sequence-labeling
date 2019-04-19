import sys
import os

import pytorch.main as pyt
import dynet_sequence_labeling.dynet_sequence_labeling as dysl

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

def load_embeddings(data_root, languages):
    return {l:Embedding.load(data_root + (f"embeddings/{l}.tar.bz2")) for l in languages}

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
    pass

def experiment_to_str(config, results):
    separator = ","
    return config_to_str(config,separator)+separator+results_to_str(results, separator)+"\n"

frameworks = ["dynet", "pytorch", "tensorflow"]
languages = ["de", "nl", "ja", "en", "sv", "zh", "sk", "ar", "el"]
tasks = ["pos", "ner"]
models = [False, True]
seeds = [613321, 5123, 421213, 521403, 322233]
batch_sizes = [1, 8, 32]
epochs = [1, 5, {"max": 50, "patience": 3}]
data_root = "../data/"

configurations = product(frameworks, seeds, batch_sizes, epochs, tasks, models, languages)
config_count = len(frameworks) * len(seeds) * len(batch_sizes) * len(epochs) * len(tasks) * len(models) * len(languages)
embeddings = load_embeddings(data_root, languages)

print(f"Progress - framework language task crf seed batchsize epochs patience")

count = 0
for framework, seed, batch_size, epoch, task, model, language in configurations:
    config = {
            "framework": framework,
            "language": language,
            "task": task,
            "crf": model,
            "seed": seed,
            "batch_size": batch_size,
            "epochs": epoch if not isinstance(epoch, dict) else epoch["max"],
            "patience": None if not isinstance(epoch, dict) else epoch["patience"],
            "hidden_size": 100,
            "dropout": 0.5,
            "data_root": data_root,
            "embedding": embeddings[language],
            "optimizer": "sgd",
            "learning_rate": 0.1,
            }
    count += 1
    print(f"{count} / {config_count} - {config_to_str(config, ' ')}")
    results = run_experiment(config)
    validate_results(results)
    with open("out.csv", "a", encoding = "utf-8") as of:
        of.write(experiment_to_str(config, results))
