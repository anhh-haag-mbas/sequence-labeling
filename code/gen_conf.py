import sys
import os
import subprocess
import time
import signal

import numpy as np
from itertools import product

if len(sys.argv) not in [2, 3, 4]:
    print("usage: gen_conf.py output [already_run_log] [split_into_x_files]")
    exit(1)

def output_file():
    return sys.argv[1]

def already_run_file():
    return sys.argv[2] if len(sys.argv) > 2 else None

def split_count():
    return int(sys.argv[3]) if len(sys.argv) > 3 else None

tasks       = ["pos", "ner"]
seeds       = [613321, 5123, 421213, 521403, 322233]
models      = [False, True]
epochs      = [{"max": 50, "patience": 3}, 5, 1]
languages   = ["da", "no", "ru", "hi", "ur", "ja", "ar"]
data_root   = "../data/"
frameworks  = ["dynet", "pytorch", "tensorflow"]
batch_sizes = [32, 8, 1]

def make_configs(params):
    converted = []
    for framework, seed, batch_size, epoch, task, model, language in params:
        converted += [[
                        "python3",
                        "slave.py",
                        framework,
                        language,
                        task,
                        str(model),
                        str(seed),
                        str(batch_size),
                        str(epoch if not isinstance(epoch, dict) else epoch["max"]),
                        str(None if not isinstance(epoch, dict) else epoch["patience"])
                     ]]
    return converted

all_configurations = make_configs(product(frameworks, seeds, batch_sizes, epochs, tasks, models, languages))

f, s, b        = len(frameworks), len(seeds), len(batch_sizes)
e, t, m, l     = len(epochs), len(tasks), len(models), len(languages)
config_count   = f * s * b * e * t * m * l

def save_configs(configs, output_f):
    with open(output_f, "w") as f:
        for config in configs:
            f.write(repr(config) + "\n")

def load_already_run():
    configs = []
    with open(already_run_file(), "r") as f:
        for line in f:
            config = line.split(" == ")[0].split(", ")
            configs += [config]
    return configs

def filter_out_already_run(configs):
    print("Filtering out already run")
    already_run = load_already_run()
    filtered = []
    for config in configs:
        if config not in already_run:
            filtered += [config]
    return filtered

cs = all_configurations
if already_run_file() is not None:
    cs = filter_out_already_run(cs)
if split_count() is not None:
    for c, i in zip(np.array_split(cs, split_count()), range(split_count())):
        save_configs(c, output_file() + str(i))
else:
    save_configs(cs, output_file())
