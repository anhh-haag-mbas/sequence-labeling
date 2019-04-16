from extractor import read_conllu_metadata, read_bio_metadata

root_path = "../../data"
languages = ["ar", "de", "el", "en", "fr", "hu", "ja", "ko", "ru", "sk", "tr"]
files = ["training", "testing", "validation"]
tasks = ["ner"]

for lang in languages[0:1]:
    for f in files:
        for task in tasks:
            data = ""
            if task == "ner":
                data = read_bio_metadata(f'{root_path}/{task}/{lang}/{f}.bio')
            elif task == "pos":
                data = read_conllu_metadata(f'{root_path}/{task}/{lang}/{f}.conllu')
            else:
                raise ValueError("Unknown task")
            print(f"{lang}\t{task}\t{f}\t{data}")
