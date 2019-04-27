import sys
import os
from itertools import product

start = "time java -Xmx6G -cp marmot.jar"
annotator_class = "marmot.morph.cmd.Annotator"
trainer_class = "marmot.morph.cmd.Trainer"
data_root = "../../data"

tasks = ["ner", "pos"]
seeds = [613321, 5123, 421213, 521403, 322233]
epochs = [1, 5, 50] 


with open("../../data/languages.txt", "r", encoding = "utf-8") as fl,\
     open("tag.sh", "w") as fa, \
     open("train.sh", "w") as ft:
    for line in fl:
        lang = line.split("-")[0].strip()
        for task, seed, epoch in product(tasks, seeds, epochs):
            file_end = 'conllu' if task == 'pos' else 'bio'
            file_root = f"{data_root}/{task}/{lang}"
            test_file = f"--test-file form-index=0,tag-index=1,{file_root}/testing.{file_end}"

            val_file = f"--test-file form-index=0,tag-index=1,{file_root}/validation.{file_end}"

            train_file = f"--train-file form-index=0,tag-index=1,{file_root}/training.{file_end}"
            file_name_prefix = f"{lang}_{task}_{seed}_{epoch}"
            pred_file = f"--pred-file out/{file_name_prefix}.out" 
            model_file = f"--model-file models/{file_name_prefix}.marmot"

            optimize = "--optimize-num-iterations true" if epoch == 50 else ""

            fa.write(f"echo {lang} - {task} - {seed} - {epoch}\n")
            fa.write(start + f" {annotator_class} {model_file} {test_file} {pred_file}\n")
            ft.write(f"echo {lang} - {task} - {seed} - {epoch}\n")
            ft.write(start + f" {trainer_class} {train_file} --seed {seed} --num-iterations {epoch} {optimize} --tag-morph false {val_file} {model_file}\n")
