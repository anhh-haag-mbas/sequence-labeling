import sys
import ipdb

#ID, FORM, LEMMA, UPOS, XPOS, FEATS, HEAD, DEPREL, DEPS, MISC = 0, 1, 2, 3, 4, 5, 6, 7, 8, 9
FORM, UPOS = 0, 1
def read_conllu(filepath):
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

def read_fasttext(filepath):
    word2embedding = dict()
    with open(filepath, 'r') as embedding_file:
        word_count, embedding_length = embedding_file.readline().split()
        for line in embedding_file:
            line = line.split()
            word2embedding[line[0]] = line[1:]

    return (word2embedding, word_count)


