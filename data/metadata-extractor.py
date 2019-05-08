import sys
from collections import defaultdict
from polyglot import Embedding

if len(sys.argv) < 2:
    raise ValueError("No filepath given")

filepaths = sys.argv[1:]

label_sets = []
output = []

for i, filepath in enumerate(filepaths):
    tokens = defaultdict(int)
    labels = defaultdict(int)
    distinct = set()

    with open(filepath, "r", encoding = "utf-8") as f:
        current = 0
        for line in f:
            if line.startswith("#"): continue
            if line.isspace(): 
                current += 1
            else: 
                split = line.split("\t")
                word = split[0].strip()
                label = split[1].strip()
                if label == "_": 
                    continue # Ignore cases where the token is unlabelled, as happens for contractions
                tokens[current] += 1
                labels[label] += 1
                distinct.add(word)

    keys = list(labels.keys())
    keys.sort()

    label_sets.append(keys)

    if len(labels) == 0: print(filepath)

    values = [filepath,len(tokens), sum(tokens.values()), len(distinct), sum(tokens.values()) / len(tokens)]
    values = values + keys + list(map(lambda k: labels[k], keys))
    values = map(str, values)
    output.append(",".join(values))
 

print(",".join(["file", "sentences", "tokens", "distinct tokens", "average tokens"] ))
for o in output:
    print(o)
