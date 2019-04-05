import sys
import os

framework = ["dynet", "pytorch", "tensorflow"]
languages = ["de", "en", "zh", "ja", "ar", "el", "sk", "", ""]
tasks = ["pos", "ner"]
crf = [True, False]
seeds = [613321, 5123, 421213, 521403, 322233]
batch_sizes = [1, 8, 32]
epochs = [1, 5, "patience-3"]

parameters = {
        "framwork": framework,
        "languages": languages,
        "tasks": tasks,
        "crf": crf,
        "seeds": seeds,
        "batch_sizes": batch_sizes,
        "epochs": epochs
        }

def config_to_str(config, separator):
    values = []
    keys = ["framework", "language", "task", "crf", "seed", "batch_size", "epochs"]
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
    pass

def run_experiment_pytorch(config):
    pass

def run_experiment_tensorflow(config):
    pass

def configuration_count(parameters):
    lengths = [len(parameters[k]) for k in parameters]
    total = 1
    for l in lengths:
        total *= l
    return total

count = 0
total = configuration_count(parameters)
for framework, seed, batch_size, epoch, task, model, language \
                in zip(epochs, seeds, batch_sizes, epochs, tasks, crf, languages):
        config = {
                "framework": framework,
                "language": language,
                "task": task,
                "crf": model,
                "seed": seed,
                "batch_size": batch_size,
                "epochs": epoch
                }
        results = run_experiment(config)
        count += 1
        print(f"{count} / {total} - {config_to_str(config, ' ')} ", file=sys.stderr)
#        with open("out.csv", "w", encoding = "utf-8") as of:
#            of.write(config_to_str(config,separator)+separator+results_to_str(results, separator))


