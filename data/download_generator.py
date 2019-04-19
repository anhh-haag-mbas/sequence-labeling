"""
Reads the language codes in languages.txt to generate bash scripts for downloading all the appropriate NER, POS and Polyglot files
"""
import sys

language_codes = []

with open("languages.txt", "r") as f, \
     open("embeddings/download_files.sh", "w") as embf, \
     open("ner/download_files.sh", "w") as nerf, \
     open("pos/download_all_files.sh", "w") as posf:

    posf.write('wget -nc -O ud-treebanks-v2.3.tgz "https://lindat.mff.cuni.cz/repository/xmlui/bitstream/handle/11234/1-2895/ud-treebanks-v2.3.tgz?sequence=1&isAllowed=y"\n')
    posf.write("tar -xzf ud-treebanks-v2.3.tgz\n")


    for line in f:
        code, language, pos = line.split("-")
        code, language, pos = code.strip(), language.strip(), pos.strip()
        language_codes.append(code)
        nerf.write(f"wget -nc -O ./{code}.tar.gz https://blender04.cs.rpi.edu/~panx2/wikiann/data/{code}.tar.gz\n")
        embf.write(f"wget -nc -O ./{code}.tar.bz2 http://polyglot.cs.stonybrook.edu/~polyglot/embeddings2/{code}/embeddings_pkl.tar.bz2\n")
        posf.write(f"cp -r ud-treebanks-v2.3/UD_{language}-{pos}/ {code}/\n")
        posf.write(f"cp {code}/*-train.conllu {code}/training.conllu\n")
        posf.write(f"cp {code}/*-dev.conllu {code}/validation.conllu\n")
        posf.write(f"cp {code}/*-test.conllu {code}/testing.conllu\n")
        posf.write(f"cat {code}/*.conllu > {code}/combined.conllu\n")
    
    for code in language_codes:
        nerf.write(f"tar -xzf {code}.tar.gz\n")

    nerf.write("python3 ner_wikidata_converter.py")
    posf.write("python3 pos_uddata_converter.py")
    for code in language_codes:
        nerf.write(f" {code}")
        posf.write(f" {code}")
    nerf.write("\n")
    posf.write("\n")
