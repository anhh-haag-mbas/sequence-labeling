from extractor import read_conllu_metadata

root_path = "../../data"
languages = ["da", "sk"]
files = ["training.conllu", "testing.conllu", "validation.conllu"]
tasks = ["pos"]

for lang in languages:
    for f in files:
        for task in tasks:
            print(f"{lang}\t{task}\t{f}\t{read_conllu_metadata(f'{root_path}/{task}/{lang}/{f}')}")
        
