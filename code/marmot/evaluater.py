import sys
from itertools import product

def compare(file_expected, file_actual, vocabulary):
    evaluation = {} 
    word_idx = 0
    tag_idx = 1
    tag_idx_conllu = 5

    words = 0
    errors = 0
    oov_errors = 0
    oov_words = 0
    with open(file_expected) as f_e,\
         open(file_actual) as f_a:
        for line_expected in f_e:
            line_actual = f_a.readline()
            if line_expected.isspace(): continue


            split_expected = line_expected.split("\t")
            split_actual = line_actual.split("\t")

            word = split_expected[word_idx].strip()
            expected = split_expected[tag_idx].strip()
            actual = split_actual[tag_idx_conllu].strip()

            #print(word, expected, actual)

            if expected not in evaluation:
                evaluation[expected] = {}

            if actual not in evaluation[expected]:
                evaluation[expected][actual] = 0

            evaluation[expected][actual] += 1
            words += 1

            if expected != actual: errors += 1

            if word not in vocabulary:
                oov_words += 1
                if expected != actual: oov_errors += 1
              
    return evaluation, words, errors, oov_words, oov_errors

def vocabulary(iterator, word_idx):
    vocab = set()
    for item in iterator:
        if item.isspace(): continue
        word = item.split("\t")[word_idx].strip()
        vocab.add(word)
    return vocab

data_root = "../../data"

tasks = ["ner", "pos"]
seeds = [613321, 5123, 421213, 521403, 322233]
epochs = [1, 5, 50] 

file_actual = "out/da_ner_5123_1.out"
file_expected = "../../data/ner/da/testing.bio"
file_training = "../../data/ner/da/training.bio"
with open(file_training) as train_file:
    vocab = vocabulary(train_file, 0)
print("words - errors - oov - oov errors")
print(compare(file_expected, file_actual, vocab))



#with open(f"{data_root}/languages.txt", "r", encoding = "utf-8") as fl:
#    for lang_line, task, seed, epoch in product(fl, tasks, seeds, epochs):
#        lang = lang_line.split("-")[0].strip()
#        file_end = "bio" if task == "ner" else "conllu"
#
#        expected_file = f"{data_root}/{task}/{lang}/testing.{file_end}"
#        training_file = f"{data_root}/{task}/{lang}/training.{file_end}"
#        actual_file = f"out/{lang}_{task}_{seed}_{epoch}.out"
#        
#        vocab = None
#        with open(training_file) as train_file:
#            vocab = vocabulary(train, 0)
#
#        print(f"{lang} - {task} - {seed} - {epoch}")
#        print("words - errors - oov - oov errors")
#        print(compare(expected_file, actual_file, vocab))
#
