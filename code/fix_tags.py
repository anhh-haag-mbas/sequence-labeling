import sys
from itertools import product

if(len(sys.argv) != 3):
    print("usage: fix_tags.py <input> <output>")
    exit(1)

in_file = sys.argv[1]
out_file = sys.argv[2]

def save(items):
    with open(out_file, 'w') as f:
        for item in items:
            f.write(str(item) + "\n")

def app(item):
    with open(out_file, 'a') as f:
        f.write(str(item) + "\n")

def load_lang_tags():
    lang_tags = {}
    with open('tags.csv') as f:
        for line in f:
            line = line.strip()
            if line == "": continue
            values = line.split(",")

            task, lang, filename = values[0].split("/")
            fit_type, _ = filename.split(".")
            if fit_type != "training": continue
            values = values[5:]
            tags = values[:(int(len(values)/2))]

            lang_tags[f"{task}-{lang}"] = tags
    return lang_tags

lang_tags = load_lang_tags()

def parse_matrix(line):
    rest = line.strip().split(",")[:15]
    framework, lang, task = rest[:3]
    matrix_values = line.strip().split(",")[15:]
    tags = lang_tags[f"{task}-{lang}"]

    matrix = {}
    keys = list(tags)
    if framework == "tensorflow":
        keys += ["<PAD>", "<UNK>"]
    keys.sort()
    if not len(keys)**2 == len(matrix_values):
        print(f"{framework}-{task}-{lang}")
        print(f"{len(keys)**2} == {len(matrix_values)}")
    assert len(keys)**2 == len(matrix_values)
    for (expected, actual), count in zip(product(keys, keys), matrix_values):
        if expected not in matrix: matrix[expected] = {}
        matrix[expected][actual] = count

    return rest, matrix

def parse_and_fix_matrix(line):
    rest, matrix = parse_matrix(line)
    framework, lang, task = rest[:3]
    if task == "ner":
        all_keys = ['B-LOC', 'B-ORG', 'B-PER', 'I-LOC', 'I-ORG', 'I-PER', 'O',
            '<PAD>', '<UNK>']
    if task == "pos":
        all_keys= ['ADJ', 'ADP', 'ADV', 'AUX', 'CCONJ', 'DET', 'INTJ', 'NOUN',
            'NUM', 'PART', 'PRON', 'PROPN', 'PUNCT', 'SCONJ', 'SYM', 'VERB',
            'X', '<PAD>', '<UNK>']
    all_keys.sort()

    for expected, actual in product(all_keys, all_keys):
        if expected not in matrix: matrix[expected] = {}
        if actual not in matrix[expected]: matrix[expected][actual] = 0

    if framework == "tensorflow":
        # total_values
        rest[8] = 0
        # total_errors
        rest[9] = 0
        # # total_oov
        # rest[10] = "-1"
        # # total_oov_errors
        # rest[11] = "-1"
        for expected, actual in product(all_keys, all_keys):
            if expected == "<PAD>": continue
            count = int(matrix[expected][actual])
            # total_values
            rest[8] += count
            if expected != actual:
                # total_errors
                rest[9] += count
        rest[8] = str(rest[8])
        rest[9] = str(rest[9])

    del matrix["<UNK>"]
    del matrix["<PAD>"]
    for key in matrix.keys():
        del matrix[key]["<UNK>"]
        del matrix[key]["<PAD>"]

    return rest, matrix

def evaluation_to_values(evaluation):
    keys = list(evaluation.keys())
    keys.sort()
    values = []
    for expected, actual in product(keys, keys):
        values.append(str(evaluation[expected][actual]))
    return values

# save(load_lang_tags().items())
# app("")
# found_dy = False
# found_tf = False
fixed_lines = []
for line in open(in_file, "r"):
    # before_matrix = parse_matrix(line)
    rest, matrix = parse_and_fix_matrix(line)
    # if rest[0] == "dynet" and rest[1] == "hi" and rest[2] == "pos" and not found_dy:
        # found_dy = True
    # elif rest[0] == "tensorflow" and rest[1] == "hi" and rest[2] == "pos" and not found_tf:
        # found_tf = True
    # else:
        # continue
    # app(rest)
    # app(before_matrix)
    # app(matrix)
    # app("")
    # app(line.strip())
    fixed_lines += [",".join(rest + evaluation_to_values(matrix))]
    # app("")
save(fixed_lines)








