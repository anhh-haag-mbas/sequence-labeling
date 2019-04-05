import sys
import os

framework = ["dynet", "pytorch", "tensorflow"]
languages = ["de", "en", "zh", "ja", "ar", "el", "sk", "", ""]
tasks = ["pos", "ner"]
crf = [True, False]
seeds = [613321, 5123, 421213, 521403, 322233]
batch_sizes = [1, 8, 32]
epochs = [1, 5, "patience-3"]

def config_to_str(config, separator):
    values = []
    keys = ["framework", "language", "task", "crf", "seed", "batch_size", "epochs"]
    values = map(lambda k: str(config[k]), keys)
    return separator.join(values) 

def results_to_str(results, separator):
    return ""

config = {
        "framework": "dynet",
        "language": "en",
        "task": "pos",
        "crf": True,
        "seed": 5124,
        "batch_size": 8,
        "epochs": 5
        }

print(config_to_str(config, ","))

#with open("out.csv", "w", encoding = "utf-8") as of:
#    of.write("Hello")
