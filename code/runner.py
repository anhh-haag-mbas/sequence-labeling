import sys
sys.path.insert(0, "./dynet")
import os
from itertools import product
import dynet_sequence_labeling

frameworks = ["dynet", "pytorch", "tensorflow"]
languages = ["de", "nl", "ja", "en", "sv", "zh", "sk", "ar", "el"] 
tasks = ["pos", "ner"]
models = [True, False]
seeds = [613321, 5123, 421213, 521403, 322233]
batch_sizes = [1, 8, 32]
epochs = [1, 5, {"max": 50, "patience": 3}]

def config_to_str(config, separator):
    values = []
    keys = ["framework", "language", "task", "crf", "seed", "batch_size", "epochs", "patience"]
    values = map(lambda k: str(config[k]), keys)
    return separator.join(values) 

def results_to_str(results, separator):
    return ""

def run_experiment(config):
    framework = config["framework"]
    if framework == "dynet":
        return run_experiment_dynet(config)
    if framework == "pytorch":
        return run_experiment_pytorch(config)
    if framework == "tensorflow":
        return run_experiment_tensorflow(config)

def run_experiment_dynet(config):
    dynet_sequence_labeling.run_experiment(config)

def run_experiment_pytorch(config):
    pass

def run_experiment_tensorflow(config):
    pass

count = 0
configurations = product(frameworks, seeds, batch_sizes, epochs, tasks, models, languages)
config_count = len(frameworks) * len(seeds) * len(batch_sizes) * len(epochs) * len(tasks) * len(models) * len(languages) 

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
            "dropout": 0.5
            }
    print(f"{config_to_str(config, ' ')}")
    results = run_experiment(config)
    count += 1
    print(f"{count} / {config_count}", file=sys.stderr)
    with open("out.csv", "w", encoding = "utf-8") as of:
        of.write(config_to_str(config,separator)+separator+results_to_str(results, separator))


