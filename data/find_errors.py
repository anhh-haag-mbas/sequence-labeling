import sys
from itertools import product

tasks       = ["pos", "ner"]
seeds       = [613321, 5123, 421213, 521403, 322233]
models      = [True, ]
epochs      = [50, 5, 1]
languages   = ["da", "no", "ru", "hi", "ur", "ja", "ar"]
frameworks  = ["dynet", "pytorch", "tensorflow"]
batch_sizes = [1, 8, 32]


configs = set()

with open("out.csv") as f:
    for line in f:
        split = line.split(",")
        framework = split[0]
        language = split[1]
        task = split[2]
        model = True if split[3] == "True" else False
        seed = int(split[4])
        batch_size = int(split[5])
        epoch = int(split[6])

        configs.add((framework, language, task, model, seed, batch_size, epoch))

expected_configs = product(frameworks, languages, tasks, models, seeds, batch_sizes, epochs)

for config in expected_configs:
    if config not in configs:
        print(config)



