import sys
import os
from itertools import product
import dynet_sequence_labeling.dynet_sequence_labeling
from polyglot.mapping import Embedding

def config_to_str(config, separator):
    keys = ["framework", "language", "task", "crf", "seed", "batch_size", "epochs", "patience"]
    values = map(lambda k: str(config[k]), keys)
    return separator.join(values)

def evaluation_to_str(evaluation, separator):
    return ""

def results_to_str(results, separator):
    keys = ["total_values", "total_errors", "total_oov", "total_oov_errors", "training_time", "evaluation_time", "epochs_run"]
    values = map(lambda k: str(config[k]), keys)
    return separator.join(values) + evaluation_to_str(results["evaluation_matrix"], separator)

def load_embeddings(data_root, languages):
    return [Embedding.load(data_root + f"embeddings/{l}.tar.bz2") for l in languages]

def run_experiment(config):
    framework = config["framework"]
    if framework == "dynet":
        return dynet_sequence_labeling.run_experiment(config)
    if framework == "pytorch":
        return run_experiment_pytorch(config)
    if framework == "tensorflow":
        return run_experiment_tensorflow(config)

def run_experiment_pytorch(config):
    pass

def run_experiment_tensorflow(config):
    pass

def experiment_to_str(config, results):
    separator = ","
    return config_to_str(config,separator)+separator+results_to_str(results, separator)

frameworks = ["dynet", "pytorch", "tensorflow"]
languages = ["de", "nl", "ja", "en", "sv", "zh", "sk", "ar", "el"]
tasks = ["pos", "ner"]
models = [True, False]
seeds = [613321, 5123, 421213, 521403, 322233]
batch_sizes = [1, 8, 32]
epochs = [1, 5, {"max": 50, "patience": 3}]
data_root = "../data/"

count = 0
configurations = product(frameworks, seeds, batch_sizes, epochs, tasks, models, languages)
config_count = len(frameworks) * len(seeds) * len(batch_sizes) * len(epochs) * len(tasks) * len(models) * len(languages)


embeddings = load_embeddings(data_root, languages)

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
            "optimizer": "sgd",
            "learning_rate": 0.1,
            "data_root": data_root,
            "embedding": polyglot_embedding
            }
    count += 1
    print(f"{count} / {config_count} - {config_to_str(config, ' ')}")

    results = run_experiment(config)
    with open("out.csv", "w", encoding = "utf-8") as of:
        of.write(experiment_to_str(config, results))
