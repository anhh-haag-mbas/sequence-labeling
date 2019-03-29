import sys
import ipdb 
from collections import defaultdict
from extractor import read_conllu
from bi_lstm_model import BiLstmModel
from bi_lstm_crf_model import BiLstmCrfModel
from helper import time, flatten

path_root = "../../../data/pos/"
train_inputs, train_labels = read_conllu(path_root + "da/training.conllu")
val_inputs, val_labels = read_conllu(path_root + "da/validation.conllu")
#embedding, word_count = read_fasttext("embeddings/cc.da.300.vec")

tags = list(set(flatten(train_labels)))
tags.sort()
vocab = list(set(flatten(train_inputs)))
vocab.sort()

int2word = ["<UNK>"] + vocab
word2int = {w:i for i, w in enumerate(int2word)}
int2tag  = tags
tag2int  = {w:i for i, w in enumerate(int2tag)}

def to_input(word, unknown = 0):
    """
    Transforms words to their respective integer representation or unknown if not in dict
    """
    if word in word2int.keys():
        return word2int[word]
    return unknown 

VOCAB_SIZE = len(int2word)
EMBED_SIZE = 86
HIDDEN_DIM = 16
OUTPUT_DIM = len(tags)

train_inputs = [[word2int[w] for w in ws] for ws in train_inputs] 
train_labels = [[tag2int[t] for t in ts] for ts in train_labels]

val_inputs = [[to_input(w) for w in ws] for ws in val_inputs]
val_labels = [[tag2int[t] for t in ts] for ts in val_labels]

test_iterations = 1
data = [defaultdict(defaultdict) for _ in range(test_iterations)]
train_output = "train_result"
train_time  = "train_time"
evaluation = "eval"

def evaluate(tagger, inputs, labels):
    evaluation = {t:0 for t in range(len(tags))}
    predictions = [tagger.predict(i) for i in inputs]
    for s_preds, s_labs in zip(predictions, labels):
        for pred, label in zip(s_preds, s_labs):
            if pred != label: evaluation[pred] += 1
    return evaluation

for testnum in range(test_iterations):
    taggers = []
    taggers.append(BiLstmModel(
                        vocab_size = VOCAB_SIZE, 
                        output_size = OUTPUT_DIM, 
                        embed_size = EMBED_SIZE, 
                        hidden_size = HIDDEN_DIM))

#    taggers.append(BiLstmCrfModel(
#                        vocab_size = VOCAB_SIZE, 
#                        output_size = OUTPUT_DIM, 
#                        embed_size = EMBED_SIZE, 
#                        hidden_size = HIDDEN_DIM))
#

    testdata = data[testnum]

    # print("Training...")
    results = [time(tagger.fit, train_inputs, train_labels) for tagger in taggers]
    for tagger, (result , elapsed) in zip(taggers, results):
        testdata[tagger.name][train_output] = result
        testdata[tagger.name][train_time] = elapsed

    for tagger in taggers: 
        evals = evaluate(tagger, val_inputs, val_labels)
        evals = {int2tag[i]:evals[i] for i in evals.keys()}
        testdata[tagger.name][evaluation] = evals

total_miss_predictions = {tagger.name: 0 for tagger in taggers}
total_training_time = {tagger.name: 0 for tagger in taggers}
total_labels = sum([len(ts) for ts in val_labels])

for testnum in range(len(data)):
    print(f"Test {testnum}")
    for tagger in data[testnum]:
        print(f"{tagger} tagger:")
        for key in data[testnum][tagger]:
            print(f"{key} = {data[testnum][tagger][key]}")
        miss_predictions = sum(data[testnum][tagger][evaluation].values())
        print(f"Total miss predictions: {miss_predictions} ~ {miss_predictions/total_labels}")

        total_training_time[tagger] += data[testnum][tagger][train_time]
        total_miss_predictions[tagger] += miss_predictions
    print("\n")


for tagger in total_training_time:
    print(f"{tagger} tagger:")
    print(f"average_training_time: {total_training_time[tagger]/test_iterations}")
    print(f"average_miss_predictions: {total_miss_predictions[tagger]/test_iterations} ~ {1 - (total_miss_predictions[tagger]/total_labels)}")

    

    

    

    

    
