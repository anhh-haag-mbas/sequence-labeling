import sys
from collections import defaultdict

if len(sys.argv) < 2:
    raise ValueError("No filepath given")

filepaths = sys.argv[1:]

longest_label_set = []
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

    if len(labels) > len(longest_label_set):
        longest_label_set = labels

    values = [filepath,len(tokens), sum(tokens.values()), len(distinct), sum(tokens.values()) / len(tokens)]
    output.append({"values": values, "labels": labels})
 

keys = list(longest_label_set.keys())
keys.sort()

print(",".join(["file", "sentences", "tokens", "distinct tokens", "average tokens"] + keys ))
for o in output:
    labels = map(lambda k: o["labels"][k], keys)
    labels = list(map(str, labels))

    values = list(map(str, o["values"]))
    values += labels
    print(",".join(values))
