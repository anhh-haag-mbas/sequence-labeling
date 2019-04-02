"""
Script for creating training, validation, and test file from unsplit bio file. 
Arguments are language codes
"""
import sys
import os

if len(sys.argv) < 2:
    print ("Run with language code")
    exit(0)

training = 0.5
validation = 0.2
test = 1 - training - validation

languages = sys.argv[1:]
for lang in languages:
    if not os.path.exists(f"{lang}"):
        os.mkdir(f"{lang}")

    sentences = []
    with open(f"wikiann-{lang}.bio", "r", encoding = 'utf-8') as fi:
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
