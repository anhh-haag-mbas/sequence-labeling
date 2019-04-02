import sys
from collections import defaultdict

ID, FORM, LEMMA, UPOS, XPOS, FEATS, HEAD, DEPREL, DEPS, MISC = 0, 1, 2, 3, 4, 5, 6, 7, 8, 9
#FORM, UPOS = 0, 1
def read_input_label_file(filepath, input_idx, label_idx):
    inputs, labels = [], []
    with open(filepath, 'r', encoding = 'utf-8') as training_file:
        current_input, current_label = [], []
        for line in training_file:
            if line.startswith("#"): continue
            if line.isspace(): # New sentence
                inputs.append(current_input)
                labels.append(current_label)
                current_input, current_label = [], []
                continue

            split = line.split()
            current_input.append(split[input_idx])
            current_label.append(split[label_idx])

        if len(current_input) > 0:
            inputs.append(current_input)
            labels.append(current_label)
    
    return (inputs, labels)

def read_conllu(filepath):
    return read_input_label_file(filepath, FORM, UPOS)

def read_bio(filepath):
    return read_input_label_file(filepath, 0, 1)

def read_fasttext(filepath):
    word2embedding = {}
    with open(filepath, 'r', encoding = 'utf-8') as embedding_file:
        word_count, _ = map(int, embedding_file.readline().split())
        for line in embedding_file:
            line = line.split(' ')
            word2embedding[line[0]] = map(float, line[1:])

    return (word2embedding, word_count)

def read_polyglot(filepath):
    pass

def read_conllu_metadata(filepath):
    _, labels = read_input_label_file(filepath, FORM, UPOS)
    sentences = len(labels)
    tags = defaultdict(int)
    for sentence in labels:
        for label in sentence:
            tags[label] += 1

    tokens = sum(tags.values())
    keys = list(tags.keys())
    keys.sort()
    label_num = len(keys)

    values = []
    for key in keys:
        values.append(str(tags[key]))

    values_string = '\t'.join(values)
    return f"{sentences}\t{tokens}\t{label_num}\t{values_string}"
            



    

