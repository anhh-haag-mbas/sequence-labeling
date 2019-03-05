import sys
import ipdb

ID, FORM, LEMMA, UPOS, XPOS, FEATS, HEAD, DEPREL, DEPS, MISC = 0, 1, 2, 3, 4, 5, 6, 7, 8, 9

def get_data(filepath):
    inputs, labels, pos_tags, words = [], [], set(), set()
    with open(filepath, 'r') as training_file:
        current_input, current_label = [], []
        for line in training_file:
            if line.startswith("#"):
                continue
            if line.isspace():
                inputs.append(current_input)
                labels.append(current_label)
                current_input, current_label = [], []
                continue

            split = line.split()
            current_input.append(split[FORM])
            current_label.append(split[UPOS])
            pos_tags.add(split[UPOS])
            words.add(split[FORM])
   

    return (inputs, labels, list(pos_tags), list(words))

