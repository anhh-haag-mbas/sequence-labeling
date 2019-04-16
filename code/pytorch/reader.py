import numpy as np

FORM, UPOS = 1, 3
def import_data(path):
    X, y = [], []
    with open(path, "r") as f:
        for line in f.readlines():
            if line.startswith("#"):
                words, tags = [], []
                continue
            if line.isspace():
                X.append(words)
                y.append(tags)
                continue

            line = line.split("\t")
            words.append(line[FORM])
            tags.append(line[UPOS])

    return X, y
