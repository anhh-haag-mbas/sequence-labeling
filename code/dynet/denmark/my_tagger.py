import sys
#import dynet as dy
import ipdb 
from collections import defaultdict
from extractor import read_conllu, read_fasttext
from simple_pos_tagger import SimplePOSTagger
from bi_pos_tagger import BiPosTagger
from bi_pos_tagger_double import BiPosTaggerDouble
from crf_bi_pos_tagger_double import CrfBiPosTaggerDouble
from helper import time
#from gensim.models import FastText
from polyglot.mapping import Embedding

train_inputs, train_labels, tags, vocab = read_conllu("data/da-ud-train.conllu")
val_inputs, val_labels, _, _ = read_conllu("data/da-ud-dev.conllu")
#(embedding, word_count), elapsed = time(read_fasttext, "embeddings/cc.da.300.vec")
#model = FastText.load_fasttext_format('embeddings/cc.da.300.bin')
embeddings, elapsed = time(Embedding.load, "embeddings/19bSoRs")

ipdb.set_trace()

int2word = ["<UNK>"] + vocab
word2int = {w:i for i, w in enumerate(int2word)}
int2tag  = ["<START>", "<END>"] + tags
tag2int  = {w:i for i, w in enumerate(int2tag)}
UNK = word2int["<UNK>"]

VOCAB_SIZE = len(int2word)
EMBED_SIZE = 64
HIDDEN_DIM = 100
OUTPUT_DIM = len(int2tag)


train_inputs = [[word2int[w] for w in ws] for ws in train_inputs] 
train_labels = [[tag2int[t] for t in ts] for ts in train_labels]

val_inputs = [[word2int.get(w, UNK) for w in ws] for ws in val_inputs]
val_labels = [[tag2int[t] for t in ts] for ts in val_labels]

test_iterations = 10
data = [dict() for _ in range(test_iterations)]
train_output = "train_result"
train_time  = "train_time"
evaluation = "eval"

def evaluate(tagger, inputs, labels):
    evaluation = {t:0 for t in range(len(int2tag))}
    predictions = [tagger.predict(i) for i in inputs]
    for s_preds, s_labs in zip(predictions, labels):
        for pred, label in zip(s_preds, s_labs):
            if pred != label: evaluation[pred] += 1
    return evaluation



for testnum in range(test_iterations):
    taggers = []
#    taggers.append((SimplePOSTagger(
#                        vocab_size = VOCAB_SIZE, 
#                        output_size = OUTPUT_DIM, 
#                        embed_size = EMBED_SIZE, 
#                        hidden_size = HIDDEN_DIM), "Simple"))
#
#    taggers.append((BiPosTagger(
#                        vocab_size = VOCAB_SIZE, 
#                        output_size = OUTPUT_DIM, 
#                        embed_size = EMBED_SIZE, 
#                        hidden_size = HIDDEN_DIM / 2), "Bi-LSTM single"))
#
#    taggers.append((BiPosTaggerDouble(
#                        vocab_size = VOCAB_SIZE, 
#                        output_size = OUTPUT_DIM, 
#                        embed_size = EMBED_SIZE, 
#                        hidden_size = HIDDEN_DIM / 2), "Bi-LSTM double"))
#
    taggers.append((CrfBiPosTaggerDouble(
                        vocab_size = VOCAB_SIZE, 
                        output_size = OUTPUT_DIM, 
                        embed_size = EMBED_SIZE, 
                        hidden_size = HIDDEN_DIM), "CRF bi-LSTM double"))

    data[testnum] = {n:dict() for _, n in taggers}

    # print("Training...")
    #print(f"Train input {[int2word[i] for i in train_inputs[0]]} -> {[train_inputs[0]]}")
    #print(f"Train labels {[int2tag[i] for i in train_labels[0]]} -> {[train_labels[0]]}")
    results = [time(tagger.fit, train_inputs, train_labels) for tagger, _ in taggers]
    for i, (result , elapsed) in enumerate(results):
        data[testnum][taggers[i][1]][train_output] = result
        data[testnum][taggers[i][1]][train_time] = elapsed

    for tagger, name in taggers: 
        evals = evaluate(tagger, val_inputs, val_labels)
        evals = {int2tag[i]:evals[i] for i in evals.keys()}
        data[testnum][name][evaluation] = evals

total_miss_predictions = {tagger: 0 for _, tagger in taggers}
total_training_time = {tagger: 0 for _, tagger in taggers}
total_labels = sum([len(labels) for labels in val_labels])

for i in range(len(data)):
    print(f"Test {i}")
    for tagger in data[i]:
        print(f"{tagger} tagger:")
        for key in data[i][tagger]:
            print(f"{key} = {data[i][tagger][key]}")
        miss_predictions = sum(data[i][tagger][evaluation].values())
        print(f"Total miss predictions: {miss_predictions}")

        total_training_time[tagger] += data[i][tagger][train_time]
        total_miss_predictions[tagger] += miss_predictions
    print("\n")

for tagger in total_training_time:
    print(f"{tagger} tagger:")
    print(f"average_training_time: {total_training_time[tagger]/test_iterations}")
    print(f"average_miss_predictions: {total_miss_predictions[tagger]/test_iterations} ~ {(total_miss_predictions[tagger]/test_iterations)/total_labels}")
    print(f"Accuracy: {1 - ((total_miss_predictions[tagger]/test_iterations)/total_labels)}") 
