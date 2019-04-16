"""
Script for creating training, validation, and test file from unsplit bio file. 
Arguments are language codes
Split data as 60% is used for training, 40% for Testing.
Out of the 60% used for training, 10% is validation
"""
import sys
import os

if len(sys.argv) < 2:
    print ("Run with language code")
    exit(0)

training = 0.5
validation = 0.1
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
        for sentence in sentences[0:training_end]:
            fo.write(sentence + "\n\n")

    with open(outfile_val, "w") as fo:
        for sentence in sentences[training_end:validation_end]:
            fo.write(sentence + "\n\n")

    with open(outfile_test, "w") as fo:
        for sentence in sentences[validation_end:test_end]:
            fo.write(sentence + "\n\n")
