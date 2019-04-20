import sys
from collections import defaultdict

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
    return read_input_label_file(filepath, 0, 1)

def read_bio(filepath):
    return read_input_label_file(filepath, 0, 1)
