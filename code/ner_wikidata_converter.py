import sys

if (len(sys.argv) < 2):
    print ("No files given")

training = 0.5
validation = 0.2
test = 1 - training - validation

# Start by removing all additional data from file
for i in range(1, len(sys.argv)):
    filename = sys.argv[i]
    
    sentences = []
    with open(filename, "r") as fi:
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
    outfile_train = filename.split(".")[0] + "_training.bio"
    outfile_val = filename.split(".")[0] + "_validation.bio"
    outfile_test = filename.split(".")[0] + "_testing.bio"

    training_end = int(sentence_count * training)
    validation_end = training_end + int(sentence_count * validation)
    test_end = sentence_count

    with open(outfile_train, "w") as fo:
        fo.write("\n\n".join(sentences[0:training_end]))

    with open(outfile_val, "w") as fo:
        fo.write("\n\n".join(sentences[training_end:validation_end]))

    with open(outfile_test, "w") as fo:
        fo.write("\n\n".join(sentences[validation_end:test_end]))
