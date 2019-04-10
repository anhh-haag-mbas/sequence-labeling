"""
Reads the language codes in languages.txt to generate bash scripts for downloading all the appropriate NER and fasttext files
"""
import sys

# Create dictionary from language codes to full language names
code2lang = dict()
with open("language_codes.txt", "r", encoding = 'utf-8') as f:
    for line in f:
        split = line.split()
        code2lang[split[0]] = "_".join(split[1:])

with open("languages.txt", "r") as f, \
     open("embeddings/download_files.sh", "w") as embf, \
     open("ner/download_files.sh", "w") as nerf:

    for line in f:
        code = line.strip()
        nerf.write(f"wget -nc -O ./{code}.tar.gz https://blender04.cs.rpi.edu/~panx2/wikiann/data/{code}.tar.gz\n")
        embf.write(f"wget -nc -O ./{code}.tar.bz2 http://polyglot.cs.stonybrook.edu/~polyglot/embeddings2/{code}/embeddings_pkl.tar.bz2\n")

with open("pos/download_all_files.sh", "w") as posf:
    posf.write('wget -nc -O ud-treebanks-v2.3.tgz "https://lindat.mff.cuni.cz/repository/xmlui/bitstream/handle/11234/1-2895/ud-treebanks-v2.3.tgz?sequence=1&isAllowed=y"')
