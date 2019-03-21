
FORM, UPOS = 1, 3
def import_data(path):
    data = []
    with open(path, "r") as f:
        for line in f.readlines():
            if line.startswith("#"):
                words, tags = [], []
                continue
            if line.isspace():
                data.append((words, tags))
                continue

            line = line.split("\t")
            words.append(line[FORM])
            tags.append(line[UPOS])

    return data
