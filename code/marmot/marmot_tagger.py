import sys
import os
import subprocess
import timeit

from itertools import product

def time(func, *args):
    start_time = timeit.default_timer()
    result = func(*args)
    elapsed = timeit.default_timer() - start_time
    return (result, elapsed)

def generate_run_strings(config):
    data_root = config["data_root"]

    lang = config["language"]
    task = config["task"]
    seed = config["seed"]
    epochs = config["epochs"]

    start = "java -Xmx6G -cp marmot.jar"
    annotator_class = "marmot.morph.cmd.Annotator"
    trainer_class = "marmot.morph.cmd.Trainer"

    file_end = 'conllu' if task == 'pos' else 'bio'
    file_root = f"{data_root}/{task}/{lang}"
    test_file = f"--test-file form-index=0,tag-index=1,{file_root}/testing.{file_end}"
    val_file = f"--test-file form-index=0,tag-index=1,{file_root}/validation.{file_end}"
    train_file = f"--train-file form-index=0,tag-index=1,{file_root}/training.{file_end}"

    file_name_prefix = f"{lang}_{task}_{seed}_{epochs}"
    pred_file = f"--pred-file {file_name_prefix}.out" 
    model_file = f"--model-file {file_name_prefix}.marmot"

    optimize = "--optimize-num-iterations true" if epochs == 50 else ""

    run_string_trainer = start + f" {trainer_class} {train_file} --seed {seed} --num-iterations {epochs} {optimize} --tag-morph false {val_file} {model_file}"
    run_string_tagger = start + f" {annotator_class} {model_file} {test_file} {pred_file}"

    return run_string_trainer, run_string_tagger

def compare(file_expected, file_actual, vocabulary):
    evaluation = {} 
    word_idx = 0
    tag_idx = 1
    tag_idx_conllu = 5

    words = 0
    errors = 0
    oov_errors = 0
    oov_words = 0
    with open(file_expected) as f_e,\
         open(file_actual) as f_a:
        for line_expected in f_e:
            line_actual = f_a.readline()
            if line_expected.isspace(): continue

            split_expected = line_expected.split("\t")
            split_actual = line_actual.split("\t")

            word = split_expected[word_idx].strip()
            expected = split_expected[tag_idx].strip()
            actual = split_actual[tag_idx_conllu].strip()

            if expected not in evaluation:
                evaluation[expected] = {}
            if actual not in evaluation[expected]:
                evaluation[expected][actual] = 0

            evaluation[expected][actual] += 1
            words += 1

            if expected != actual: errors += 1

            if word not in vocabulary:
                oov_words += 1
                if expected != actual: oov_errors += 1
              
    return evaluation, words, errors, oov_words, oov_errors

def vocabulary(iterator, word_idx):
    vocab = set()
    for item in iterator:
        if item.isspace(): continue
        word = item.split("\t")[word_idx].strip()
        vocab.add(word)
    return vocab

def evaluate(config):
    task = config["task"]
    language = config["language"]
    seed  = config["seed"]
    epochs = config["epochs"]
    data_root = config["data_root"]

    file_end = "bio" if task == "ner" else "conllu"
    file_actual = f"{language}_{task}_{seed}_{epochs}.out"
    file_expected = f"{data_root}/{task}/{language}/testing.{file_end}"
    file_training = f"{data_root}/{task}/{language}/training.{file_end}"
    with open(file_training) as train_file:
        vocab = vocabulary(train_file, 0)

    return compare(file_expected, file_actual, vocab)

def cleanup(config):
    task = config["task"]
    language = config["language"]
    seed  = config["seed"]
    epochs = config["epochs"]

    file_name_prefix = f"{language}_{task}_{seed}_{epochs}"
    os.remove(f"{file_name_prefix}.out")
    os.remove(f"{file_name_prefix}.marmot")

def run_experiment(config):
    run_string_trainer, run_string_tagger = generate_run_strings(config)

    call_results, train_time = time(lambda: subprocess.run(run_string_trainer.split()))
    call_results.check_returncode()

    call_results, tag_time   = time(lambda: subprocess.run(run_string_tagger.split()))
    call_results.check_returncode()

    evaluation, total_words, total_errors, total_oov, total_oov_errors = evaluate(config)

    cleanup(config)
    
    return {
            "total_values": total_words,
            "total_errors": total_errors,
            "total_oov": total_oov,
            "total_oov_errors": total_oov_errors,
            "training_time": train_time,
            "evaluation_time": tag_time,
            "evaluation_matrix": evaluation,
            "epochs_run": "Not available"
            }
