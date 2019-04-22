import sys
import os
import subprocess
import time
import signal

from itertools import product
from polyglot.mapping import Embedding

def load_embeddings(data_root, languages):
    return {l:Embedding.load(data_root + (f"embeddings/{l}.tar.bz2")) for l in languages}

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

tasks       = ["pos", "ner"]
seeds       = [613321, 5123, 421213, 521403, 322233]
models      = [False, True]
epochs      = [{"max": 50, "patience": 3}, 5, 1]
languages   = ["da", "no", "ru", "hi", "ur", "ja", "ar"]
data_root   = "../data/"
frameworks  = ["dynet", "pytorch", "tensorflow"]
batch_sizes = [32, 8, 1]

configurations = product(frameworks, seeds, batch_sizes, epochs, tasks, models, languages)

f, s, b        = len(frameworks), len(seeds), len(batch_sizes)
e, t, m, l     = len(epochs), len(tasks), len(models), len(languages)
config_count   = f * s * b * e * t * m * l

embeddings = load_embeddings(data_root, languages)

print(f"Progress - framework language task crf seed batchsize epochs patience")

count = 0
processes = []
max_process_count = 33

def signal_handler(sig, frame):
    print("Killing subprocesses")
    for process in processes:
        process.kill()
    sys.exit(0)
signal.signal(signal.SIGINT, signal_handler)

try:
    for framework, seed, batch_size, epoch, task, model, language in configurations:
        config = [
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
                 ]
        if len(processes) > max_process_count:
            print(f"Running the max {len(processes)} processes, now waiting...")
        while(len(processes) > max_process_count):
            time.sleep(10)
            for process in processes:
                process.poll()
                if process.returncode is not None:
                    with open("log", "a", encoding = "utf-8") as of:
                        of.write(config_to_string(config) + "\n" + str(process.returncode))
            processes = [p for p in processes if p.returncode is None]
        processes += [subprocess.Popen(config)]
        print(" ".join(config))
        count += 1
except:
    print("Killing subprocesses")
    for process in processes:
        process.kill()
    raise sys.exc_info()[0]

