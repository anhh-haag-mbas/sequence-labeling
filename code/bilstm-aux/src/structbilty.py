#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DSDS - a neural network based tagger (bi-LSTM) - re-factored tagger from https://arxiv.org/abs/1604.05529 with support for bilstm-CRF
:author: Barbara Plank
"""
import random
import time
import sys
import numpy as np
import os
import dill
import _dynet as dynet
import codecs

from lib.mnnl import init_dynet
from lib.mio import SeqData
from lib.constants import MAX_SEED 
from lib.mmappers import TRAINER_MAP, ACTIVATION_MAP, INITIALIZER_MAP, BUILDERS
from lib.argparser import init_parser
from nn_tagger import NNTagger  

def main():
    parser = init_parser()
    try:
        args = parser.parse_args()
    except:
        parser.print_help()
        exit()

    if args.train:
        if len(args.train) > 1:
            if not args.pred_layer:
                print("--pred_layer required!")
                exit()
        elif len(args.train) == 1 and not args.pred_layer:
            args.pred_layer = [args.h_layers] # assumes h_layers is 1

    if args.c_in_dim == 0:
        print(">>> disable character embeddings <<<")

    if args.minibatch_size > 1:
        print(">>> using minibatch_size {} <<<".format(args.minibatch_size))

    if args.viterbi_loss:
        if not args.crf:
            print("--crf (global decoding) needs to be active when --viterbi is used")
            exit()
    if args.crf:
        if args.viterbi_loss:
            print(">>> using global decoding (Viterbi loss) <<<")
        else:
            print(">>> using global decoding (CRF, neg-log loss) <<<")

    if args.patience:
        if not args.dev or not args.model:
            print("patience requires a dev set and model path (--dev and --model)")
            exit()

    # check if --save folder exists
    if args.model:
        if os.path.isdir(args.model):
            if not os.path.exists(args.model):
                print("Creating {}..".format(args.model))
                os.makedirs(args.model)
        elif os.path.isdir(os.path.dirname(args.model)) and not os.path.exists(os.path.dirname(args.model)):
            print("Creating {}..".format(os.path.dirname(args.model)))
            os.makedirs(os.path.dirname(args.model))

    if args.output:
        if os.path.isdir(os.path.dirname(args.output)) and not os.path.exists(os.path.dirname(args.output)):
            os.makedirs(os.path.dirname(args.output))

    if not args.seed:
        ## set seed
        seed = random.randint(1, MAX_SEED)
    else:
        seed = args.seed

    print(">>> using seed: {} <<< ".format(seed))
    np.random.seed(seed)
    random.seed(seed)

    init_dynet(seed)

    if args.mimickx_model:
        from mimickx import Mimickx, load_model  # make sure PYTHONPATH is set
        print(">>> Loading mimickx model {} <<<".format(args.mimickx_model))

    model_path = args.model

    start = time.time()

    if args.train and len( args.train ) != 0:

        tagger = NNTagger(args.in_dim,
                          args.h_dim,
                          args.c_in_dim,
                          args.c_h_dim,
                          args.h_layers,
                          args.pred_layer,
                          embeds_file=args.embeds,
                          w_dropout_rate=args.word_dropout_rate,
                          c_dropout_rate=args.char_dropout_rate,
                          activation=ACTIVATION_MAP[args.ac],
                          mlp=args.mlp,
                          activation_mlp=ACTIVATION_MAP[args.ac_mlp],
                          noise_sigma=args.sigma,
                          learning_algo=args.trainer,
                          learning_rate=args.learning_rate,
                          backprob_embeds=args.disable_backprob_embeds,
                          initializer=INITIALIZER_MAP[args.initializer],
                          builder=BUILDERS[args.builder],
                          crf=args.crf,
                          mimickx_model_path=args.mimickx_model,
                          dictionary=args.dictionary, type_constraint=args.type_constraint,
                          lex_dim=args.lex_dim, embed_lex=args.embed_lex)

        dev = None
        train = SeqData(args.train)
        if args.dev:
            dev = SeqData(args.dev)

        tagger.fit(train, args.iters,
                   dev=dev,
                   model_path=model_path, patience=args.patience, minibatch_size=args.minibatch_size, log_losses=args.log_losses)

        if not args.dev and not args.patience:  # in case patience is active it gets saved in the fit function
            save(tagger, model_path)

    if args.test and len( args.test ) != 0:

        tagger = load(args.model, args.dictionary)

        # check if mimickx provided after training
        if args.mimickx_model:
            tagger.mimickx_model_path = args.mimickx_model
            tagger.mimickx_model = load_model(args.mimickx_model)

        stdout = sys.stdout
        # One file per test ...
        if args.test:
            test = SeqData(args.test) # read in all test data

            for i, test_file in enumerate(args.test): # expect them in same order
                if args.output is not None:
                    sys.stdout = codecs.open(args.output + ".task{}".format(i), 'w', encoding='utf-8')

                start_testing = time.time()

                print('\nTesting task{}'.format(i),file=sys.stderr)
                print('*******\n',file=sys.stderr)
                correct, total = tagger.evaluate(test, "task{}".format(i),
                                                 output_predictions=args.output,
                                                 output_confidences=args.output_confidences, raw=args.raw,
                                                 unk_tag=None)
                if not args.raw:
                    print("\nTask{} test accuracy on {} items: {:.4f}".format(i, i+1, correct/total),file=sys.stderr)
                print(("Done. Took {0:.2f} seconds in total (testing took {1:.2f} seconds).".format(time.time()-start,
                                                                                                    time.time()-start_testing)),file=sys.stderr)
                sys.stdout = stdout
    if args.train:
        print("Info: biLSTM\n\t"+"\n\t".join(["{}: {}".format(a,v) for a, v in vars(args).items()
                                          if a not in ["train","test","dev","pred_layer"]]))
    else:
        # print less when only testing, as not all train params are stored explicitly
        print("Info: biLSTM\n\t" + "\n\t".join(["{}: {}".format(a, v) for a, v in vars(args).items()
                                                if a not in ["train", "test", "dev", "pred_layer",
                                                             "initializer","ac","word_dropout_rate",
                                                             "patience","sigma","disable_backprob_embed",
                                                             "trainer", "dynet_seed", "dynet_mem","iters"]]))

    tagger = load(args.model, args.dictionary)

    if args.save_embeds:
        tagger.save_embeds(args.save_embeds)

    if args.save_lexembeds:
        tagger.save_lex_embeds(args.save_lexembeds)

    if args.save_cwembeds:
        tagger.save_cw_embeds(args.save_cwembeds)

    if args.save_lwembeds:
        tagger.save_lw_embeds(args.save_lwembeds)
    
    if args.transition_matrix:
        tagger.save_transition_matrix(args.transition_matrix)

def load(model_path, local_dictionary=None):
    """
    load a model from file; specify the .model file, it assumes the *pickle file in the same location
    """
    print("load model.. ", model_path)
    myparams = dill.load(open(model_path+".params.pickle", "rb"))
    if not "mimickx_model_path" in myparams:
        myparams["mimickx_model_path"] = None
    if local_dictionary:
        myparams["path_to_dictionary"] = local_dictionary
    tagger = NNTagger(myparams["in_dim"],
                      myparams["h_dim"],
                      myparams["c_in_dim"],
                      myparams["c_h_dim"],
                      myparams["h_layers"],
                      myparams["pred_layer"],
                      activation=myparams["activation"],
                      mlp=myparams["mlp"],
                      activation_mlp=myparams["activation_mlp"],
                      builder=myparams["builder"],
                      crf=myparams["crf"],
                      mimickx_model_path=myparams["mimickx_model_path"],
                      dictionary=myparams["path_to_dictionary"],
                      type_constraint=myparams["type_constraint"],
                      lex_dim=myparams["lex_dim"],
                      embed_lex=myparams["embed_lex"]
                      )
    tagger.set_indices(myparams["w2i"],myparams["c2i"],myparams["task2tag2idx"],myparams["w2c_cache"], myparams["l2i"])
    tagger.set_counts(myparams["wcount"], myparams["wtotal"], myparams["ccount"], myparams["ctotal"])
    tagger.build_computation_graph(myparams["num_words"],
                                       myparams["num_chars"])
    tagger.model.populate(model_path+".model")
    print("model loaded: {}".format(model_path))
    return tagger

def save(nntagger, model_path):
    """
    save a model; dynet only saves the parameters, need to store the rest separately
    """
    modelname = model_path + ".model"
    nntagger.model.save(modelname)
    myparams = {"num_words": len(nntagger.w2i),
                "num_chars": len(nntagger.c2i),
                "w2i": nntagger.w2i,
                "c2i": nntagger.c2i,
                "wcount": nntagger.wcount,
                "wtotal": nntagger.wtotal,
                "ccount": nntagger.ccount,
                "ctotal": nntagger.ctotal,
                "w2c_cache": nntagger.w2c_cache,
                "task2tag2idx": nntagger.task2tag2idx,
                "activation": nntagger.activation,
                "mlp": nntagger.mlp,
                "activation_mlp": nntagger.activation_mlp,
                "in_dim": nntagger.in_dim,
                "h_dim": nntagger.h_dim,
                "c_in_dim": nntagger.c_in_dim,
                "c_h_dim": nntagger.c_h_dim,
                "h_layers": nntagger.h_layers,
                "pred_layer": nntagger.pred_layer,
                "builder": nntagger.builder,
                "crf": nntagger.crf,
                "mimickx_model_path": nntagger.mimickx_model_path,
                "path_to_dictionary": nntagger.path_to_dictionary,
                "type_constraint": nntagger.type_constraint,
                "lex_dim": nntagger.lex_dim,
                "embed_lex": nntagger.embed_lex,
                "l2i": nntagger.l2i
                }
    dill.dump(myparams, open( model_path+".params.pickle", "wb" ) )
    print("model stored: {}".format(modelname))
    del nntagger

if __name__=="__main__":
    main()
