import argparse
from lib.mmappers import TRAINER_MAP, ACTIVATION_MAP, INITIALIZER_MAP, BUILDERS

def init_parser():
    parser = argparse.ArgumentParser(description="""Run the bi-LSTM tagger""", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    group_main = parser.add_argument_group('Main', 'main arguments')
    group_main.add_argument("--model", help="path to store/load model [required]", required=True)
    group_main.add_argument("--train", nargs='*', help="path to train file [if multiple files are given actives MTL]") # allow multiple train files, each asociated with a task = position in the list
    group_main.add_argument("--dev", nargs='*', help="dev file(s)", required=False)
    group_main.add_argument("--test", nargs='*', help="test file(s) [same order as --train]", required=False)

    group_model = parser.add_argument_group('Model', 'specify model parameters')
    group_model.add_argument("--in_dim", help="input dimension", type=int, default=64) # default Polyglot size
    group_model.add_argument("--h_dim", help="hidden dimension [default: 100]", type=int, default=100)
    group_model.add_argument("--c_in_dim", help="input dimension for character embeddings", type=int, default=100)
    group_model.add_argument("--c_h_dim", help="hidden dimension for character embeddings", type=int, default=100)
    group_model.add_argument("--h_layers", help="number of stacked LSTMs [default: 1 = no stacking]", required=False, type=int, default=1)
    group_model.add_argument("--pred_layer", nargs='*', help="predict task at this layer [default: last layer]", required=False) # for each task the layer on which it is predicted (default 1)
    group_model.add_argument("--embeds", help="word embeddings file", required=False, default=None)
    group_model.add_argument("--crf", help="use CRF instead of local decoding", default=False, action="store_true")
    group_model.add_argument("--viterbi-loss", help="Use viterbi loss training (only active if --crf is on)", action="store_true", default=False)
    group_model.add_argument("--transition-matrix", help="store transition matrix from CRF")

    group_model.add_argument("--builder", help="RNN builder (default: lstmc)", choices=BUILDERS.keys(), default="lstmc")

    group_model.add_argument("--mlp", help="add additional MLP layer of this dimension [default 0=disabled]", default=0, type=int)
    group_model.add_argument("--ac-mlp", help="activation function for optional MLP layer [rectify, tanh, ...] (default: tanh)",
                        default="tanh", choices=ACTIVATION_MAP.keys())
    group_model.add_argument("--ac", help="activation function between hidden layers [rectify, tanh, ...]", default="tanh",
                             choices=ACTIVATION_MAP.keys())

    group_input = parser.add_argument_group('Input', 'specific input options')
    group_input.add_argument("--raw", help="expects raw text input (one sentence per line)", required=False, action="store_true", default=False)

    group_output = parser.add_argument_group('Output', 'specific output options')
    group_output.add_argument("--dictionary", help="use dictionary as additional features or type constraints (with --type-constraints)", default=None)
    group_output.add_argument("--type-constraint", help="use dictionary as type constraints", default=False, action="store_true")
    group_output.add_argument("--embed-lex", help="use dictionary as type constraints", default=False, action="store_true")
    group_output.add_argument("--lex-dim", help="input dimension for lexical features", default=0, type=int)
    group_output.add_argument("--output", help="output predictions to file [word|gold|pred]", default=None)
    group_output.add_argument("--output-confidences", help="output tag confidences", action="store_true", default=False)
    group_output.add_argument("--save-embeds", help="save word embeddings to file", required=False, default=None)
    group_output.add_argument("--save-lexembeds", help="save lexicon embeddings to file", required=False, default=None)
    group_output.add_argument("--save-cwembeds", help="save character-based word-embeddings to file", required=False, default=None)
    group_output.add_argument("--save-lwembeds", help="save lexicon-based word-embeddings to file", required=False, default=None)
    group_output.add_argument("--mimickx-model", help="use mimickx model for OOVs", required=False, default=None, type=str)


    group_opt = parser.add_argument_group('Optimizer', 'specify training parameters')
    group_opt.add_argument("--iters", help="training iterations", type=int,default=20)
    group_opt.add_argument("--sigma", help="sigma of Gaussian noise",default=0.2, type=float)
    group_opt.add_argument("--trainer", help="trainer [default: sgd]", choices=TRAINER_MAP.keys(), default="sgd")
    group_opt.add_argument("--learning-rate", help="learning rate [0: use default]", default=0, type=float) # see: http://dynet.readthedocs.io/en/latest/optimizers.html
    group_opt.add_argument("--patience", help="patience [default: 0=not used], requires specification of --dev and model path --save", required=False, default=0, type=int)
    group_opt.add_argument("--log-losses", help="log loss (for each task if multiple active)", required=False, action="store_true", default=False)
    group_opt.add_argument("--word-dropout-rate", help="word dropout rate [default: 0.25], if 0=disabled, recommended: 0.25 (Kiperwasser & Goldberg, 2016)", required=False, default=0.25, type=float)
    group_opt.add_argument("--char-dropout-rate", help="char dropout rate [default: 0=disabled]", required=False, default=0.0, type=float)
    group_opt.add_argument("--disable-backprob-embeds", help="disable backprob into embeddings (default is to update)",
                        required=False, action="store_false", default=True)
    group_opt.add_argument("--initializer", help="initializer for embeddings (default: constant)",
                        choices=INITIALIZER_MAP.keys(), default="constant")


    group_dynet = parser.add_argument_group('DyNet', 'DyNet parameters')
    group_dynet.add_argument("--seed", help="random seed (also for DyNet)", required=False, type=int)
    group_dynet.add_argument("--dynet-mem", help="memory for DyNet", required=False, type=int)
    group_dynet.add_argument("--dynet-gpus", help="1 for GPU usage", default=0, type=int) # warning: non-deterministic results on GPU https://github.com/clab/dynet/issues/399
    group_dynet.add_argument("--dynet-autobatch", help="if 1 enable autobatching", default=0, type=int)
    group_dynet.add_argument("--minibatch-size", help="size of minibatch for autobatching (1=disabled)", default=1, type=int)
    return parser