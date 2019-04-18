import sys
from collections import defaultdict

if len(sys.argv) < 2:
    raise ValueError("No label index given")
if len(sys.argv) < 3:
    raise ValueError("No filepath given")

filepaths = sys.argv[2:]
idx = int(sys.argv[1])

label_sets = []
output = []

for i, filepath in enumerate(filepaths):
    tokens = defaultdict(int)
    labels = defaultdict(int)

    with open(filepath, "r", encoding = "utf-8") as f:
        current = 0
        for line in f:
            if line.startswith("#"): continue
            if line.isspace(): 
                current += 1
            else: 
                split = line.split("\t")
                label = split[idx].strip()
                if label == "_": continue # Ignore cases where the token is unlabelled, as would happen for english words like cannot
                tokens[current] += 1
                labels[label] += 1

    keys = list(labels.keys())
    keys.sort()

    label_sets.append(keys)

    values = [filepath,len(tokens), sum(tokens.values()), sum(tokens.values()) / len(tokens)]
    values = values + list(map(lambda k: labels[k], keys))
    values = map(str, values)
    output.append(",".join(values))
 
longest_label_set = []
for label_set in label_sets:
    if len(label_set) > len(longest_label_set):
        longest_label_set = label_set

print(",".join(["file", "sentences", "tokens", "average tokens"] + longest_label_set))
for o in output:
    print(o)
