"""
Script for creating training, validation, and test file from unsplit bio file. 
First argument is the language directory to move the files to, second argument is the file.
"""
import sys
import os

if len(sys.argv) < 3:
    print ("Run with language code and filepath")
    exit(0)

lang = sys.argv[1]

if not os.path.exists(f"{lang}"):
    os.mkdir(f"{lang}")

training = 0.5
validation = 0.2
test = 1 - training - validation

sentences = []
with open(sys.argv[2], "r") as fi:
    sentence = []
    for line in fi:
        if line.isspace():
            sentences.append("\n".join(sentence))
            sentence = []
        else:
            split = line.split()
            sentence.append(split[0] + " " + split[-1])
    if len(sentence) != 0: 
        sentences.append(sentence)

sentence_count = len(sentences)
outfile_train = f"{lang}/training.bio" 
outfile_val = f"{lang}/validation.bio"
outfile_test = f"{lang}/testing.bio"

training_end = int(sentence_count * training)
validation_end = training_end + int(sentence_count * validation)
test_end = sentence_count

with open(outfile_train, "w") as fo:
    fo.write("\n\n".join(sentences[0:training_end]))

with open(outfile_val, "w") as fo:
    fo.write("\n\n".join(sentences[training_end:validation_end]))

with open(outfile_test, "w") as fo:
    fo.write("\n\n".join(sentences[validation_end:test_end]))
