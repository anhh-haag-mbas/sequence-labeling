"""
Script for creating training, validation, and test file from unsplit conllu file. 
Arguments are language codes
Ideal split is 4000, 500, 500 if imposible tries to split 80%, 10%, 10% with a warning
"""
import sys
import os

if len(sys.argv) < 2:
    print ("Needs a language codes as argument")
    exit(0)

# Fallback splits
training = 0.8
validation = 0.1
test = 1 - training - validation

languages = sys.argv[1:]
for lang in languages:
    sentences = []
    with open(f"{lang}/combined.conllu", "r", encoding = 'utf-8') as fi:
        sentence = []
        for line in fi:
            if line.startswith("#"): continue
            if line.isspace():
                sentences.append("\n".join(sentence))
                sentence = []
            else:
                split = line.split()
                word = split[1]
                label = split[3]
                if label == "_": continue
                sentence.append(split[1] + "\t" + split[3])

            if len(sentences) > 7000:
                break

        if len(sentence) != 0: # Add last sentence if any
            sentences.append(sentence)

    sentence_count = len(sentences)
    outfile_train = f"{lang}/training.conllu" 
    outfile_val = f"{lang}/validation.conllu"
    outfile_test = f"{lang}/testing.conllu"

    if sentence_count >= 5000:
        training_end = 4000
        validation_end = 4500
        test_end = 5000
    else:
        training_end = int(sentence_count * training)
        validation_end = training_end + int(sentence_count * validation)
        test_end = sentence_count
        print(f"Language {lang}: Only {sentence_count} < 7000 sentences, splitting {training_end}, {validation_end}, {test_end}")
 
    with open(outfile_train, "w") as fo:
        for sentence in sentences[0:training_end]:
            fo.write(sentence + "\n\n")

    with open(outfile_val, "w") as fo:
        for sentence in sentences[training_end:validation_end]:
            fo.write(sentence + "\n\n")

    with open(outfile_test, "w") as fo:
        for sentence in sentences[validation_end:test_end]:
            fo.write(sentence + "\n\n")
