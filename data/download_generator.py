"""
Reads the language codes in languages.txt to generate bash scripts for downloading all the appropriate NER and fasttext files
TODO: Somehow extend with polyglot and POS file downloads
"""
import sys

# Create dictionary from language codes to full language names
code2lang = dict()
with open("language_codes.txt", "r", encoding = 'utf-8') as f:
    for line in f:
        split = line.split()
        code2lang[split[0]] = "_".join(split[1:])

with open("languages.txt", "r") as f, \
     open("embeddings/fasttext/download_files.sh", "w") as fembf, \
     open("ner/download_files.sh", "w") as nerf:

    for line in f:
        code = line.strip()
        lang = code2lang[code]
        nerf.write(f"wget -O ./{lang}.tar.gz https://blender04.cs.rpi.edu/~panx2/wikiann/data/{code}.tar.gz\n")
        fembf.write(f"wget -O ./{lang}.bin.gz https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.{code}.300.bin.gz")
