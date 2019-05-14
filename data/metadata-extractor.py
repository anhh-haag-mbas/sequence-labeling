import sys
from collections import defaultdict
from polyglot.mapping import Embedding

if len(sys.argv) < 2:
    raise ValueError("No filepath given")

filepaths = sys.argv[1:]

longest_label_set = []
output = []

languages = ["ar", "da", "hi", "ja", "no", "ru", "ur"]

embedding = {lang: Embedding.load(f"embeddings/{lang}.tar.bz2") for lang in languages}

for i, filepath in enumerate(filepaths):
    tokens = defaultdict(int)
    labels = defaultdict(int)
    oov = 0
    distinct = set()

    lang = filepath.split("/")[1]

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

                if word not in embedding[lang]:
                    oov += 1

                tokens[current] += 1
                labels[label] += 1
                distinct.add(word)

    if len(labels) > len(longest_label_set):
        longest_label_set = labels

    values = [filepath,len(tokens), sum(tokens.values()), len(distinct),
            sum(tokens.values()) / len(tokens), oov]

    output.append({"values": values, "labels": labels})
 

keys = list(longest_label_set.keys())
keys.sort()

print(",".join(["file", "sentences", "tokens", "distinct tokens", 
    "average tokens", "oov tokens"] + keys ))


for o in output:
    labels = map(lambda k: o["labels"][k], keys)
    labels = list(map(str, labels))

    values = list(map(str, o["values"]))
    values += labels
    print(",".join(values))
