import sys

ID, FORM, LEMMA, UPOS, XPOS, FEATS, HEAD, DEPREL, DEPS, MISC = 0, 1, 2, 3, 4, 5, 6, 7, 8, 9
#FORM, UPOS = 0, 1
def read_conllu(filepath):
    inputs, labels = [], []
    with open(filepath, 'r') as training_file:
        current_input, current_label = [], []
        for line in training_file:
            if line.startswith("#"): continue
            if line.isspace(): # New sentence
                inputs.append(current_input)
                labels.append(current_label)
                current_input, current_label = [], []
                continue

            split = line.split()
            word, tag = split[FORM], split[UPOS]
            
            current_input.append(word)
            current_label.append(tag)
    
    return (inputs, labels)


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
