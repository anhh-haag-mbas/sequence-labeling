import sys



results = {}
languages = set()

with open(sys.argv[1], "r") as f:
    headers = f.readline().strip().split(",")

    for lines in f:
        split = lines.strip().split(",")

        lang = split[0][4:6]
        if lang not in languages:
            languages.add(lang)
            results[lang] = { "testing": {}, "training": {}, "validation": {} }

        file_type = split[0][7:split[0].index(".")]

        for i, value in enumerate(split):
            results[lang][file_type][headers[i]] = value


print("language, training tokens, training distinct, training avg. tokens, testing tokens, testing distinct, testing avg. tokens")
for lang in languages:
    training = results[lang]["training"]
    testing = results[lang]["testing"]

    print(lang, training["tokens"], training["distinct tokens"], training["average tokens"], testing["tokens"], testing["distinct tokens"], testing["average tokens"], sep = ",")



