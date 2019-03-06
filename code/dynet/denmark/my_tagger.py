import sys
#import dynet as dy
import ipdb 
from extractor import read_conllu, read_fasttext
from simple_pos_tagger import SimplePOSTagger
from bi_pos_tagger import BiPosTagger
from bi_pos_tagger_double import BiPosTaggerDouble
from helper import time

train_inputs, train_labels, tags, vocab = read_conllu("data/da_ddt-ud-train.conllu")
val_inputs, val_labels, _, _ = read_conllu("data/da_ddt-ud-dev.conllu")
embedding, word_count = read_fasttext("embeddings/cc.da.300.vec")

ipdb.set_trace()

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

test_iterations = 20
data = [dict() for _ in range(test_iterations)]
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
    taggers.append((SimplePOSTagger(
                        vocab_size = VOCAB_SIZE, 
                        output_size = OUTPUT_DIM, 
                        embed_size = EMBED_SIZE, 
                        hidden_size = HIDDEN_DIM), "Simple"))

    taggers.append((BiPosTagger(
                        vocab_size = VOCAB_SIZE, 
                        output_size = OUTPUT_DIM, 
                        embed_size = EMBED_SIZE, 
                        hidden_size = HIDDEN_DIM / 2), "Single LSTM bi"))

    taggers.append((BiPosTaggerDouble(
                        vocab_size = VOCAB_SIZE, 
                        output_size = OUTPUT_DIM, 
                        embed_size = EMBED_SIZE, 
                        hidden_size = HIDDEN_DIM / 2), "Double LSTM bi"))

    data[testnum] = {n:dict() for _, n in taggers}

    # print("Training...")
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
    print(f"average_miss_predictions: {total_miss_predictions[tagger]/test_iterations}")

    

    

    

    

    
