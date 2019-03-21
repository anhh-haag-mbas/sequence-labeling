# Training
python src/structbilty.py --dynet-mem 1500 --train data/da-ud-train.conllu --test data/da-ud-test.conllu --iters 10 --model da

Task0 test accuracy on 1 items: 0.9585
Done. Took 1746.70 seconds in total (testing took 3.38 seconds).
>>> using seed: 98757373 <<<
init parameters
Caching top 5265 words
17553 w features, 100 c features
build graph
train..
The dy.parameter(...) call is now DEPRECATED.
        There is no longer need to explicitly add parameters to the computation graph.
        Any used parameter will be added automatically.
iter 0   total loss: 0.79
iter 1   total loss: 0.32
iter 2   total loss: 0.24
iter 3   total loss: 0.19
iter 4   total loss: 0.17
iter 5   total loss: 0.14
iter 6   total loss: 0.12
iter 7   total loss: 0.11
iter 8   total loss: 0.10
iter 9   total loss: 0.09
model stored: da.model
load model..  da
model loaded: da
Info: biLSTM
        model: da
        in_dim: 64
        h_dim: 100
        c_in_dim: 100
        c_h_dim: 100
        h_layers: 1
        embeds: None
        crf: False
        viterbi_loss: False
        transition_matrix: None
        builder: lstmc
        mlp: 0
        ac_mlp: tanh
        ac: tanh
        raw: False
        dictionary: None
        type_constraint: False
        embed_lex: False
        lex_dim: 0
        output: None
        output_confidences: False
        save_embeds: None
        save_lexembeds: None
        save_cwembeds: None
        save_lwembeds: None
        mimickx_model: None
        iters: 10
        sigma: 0.2
        trainer: sgd
        learning_rate: 0
        patience: 0
        log_losses: False
        word_dropout_rate: 0.25
        char_dropout_rate: 0.0
        disable_backprob_embeds: True
        initializer: constant
        seed: None
        dynet_mem: 1500
        dynet_gpus: 0
        dynet_autobatch: 0
        minibatch_size: 1
load model..  da
model loaded: da

# Testing with patience
python src/structbilty.py --dynet-mem 1500 --train data/da-ud-train.conllu --dev data/da-ud-dev.conllu --test data/da-ud-test.conllu --iters 50 --model da --patience 2

Task0 test accuracy on 1 items: 0.9562
Done. Took 1776.40 seconds in total (testing took 3.54 seconds).
>>> using seed: 42050192 <<<
init parameters
Caching top 5265 words
17553 w features, 100 c features
build graph
Using early stopping with patience of 2...
train..
The dy.parameter(...) call is now DEPRECATED.
        There is no longer need to explicitly add parameters to the computation graph.
        Any used parameter will be added automatically.
iter 0   total loss: 0.78
dev accuracy: 0.8881
Accuracy 0.8881 is better than best val accuracy 0.0000.
model stored: da.model
iter 1   total loss: 0.33
dev accuracy: 0.9378
Accuracy 0.9378 is better than best val accuracy 0.8881.
model stored: da.model
iter 2   total loss: 0.25
dev accuracy: 0.9453
Accuracy 0.9453 is better than best val accuracy 0.9378.
model stored: da.model
iter 3   total loss: 0.20
dev accuracy: 0.9496
Accuracy 0.9496 is better than best val accuracy 0.9453.
model stored: da.model
iter 4   total loss: 0.17
dev accuracy: 0.9547
Accuracy 0.9547 is better than best val accuracy 0.9496.
model stored: da.model
iter 5   total loss: 0.15
dev accuracy: 0.9574
Accuracy 0.9574 is better than best val accuracy 0.9547.
model stored: da.model
iter 6   total loss: 0.13
dev accuracy: 0.9605
Accuracy 0.9605 is better than best val accuracy 0.9574.
model stored: da.model
iter 7   total loss: 0.11
dev accuracy: 0.9632
Accuracy 0.9632 is better than best val accuracy 0.9605.
model stored: da.model
iter 8   total loss: 0.10
dev accuracy: 0.9618
Accuracy 0.9618 is worse than best val loss 0.9632.
iter 9   total loss: 0.09
dev accuracy: 0.9618
Accuracy 0.9618 is worse than best val loss 0.9632.
No improvement for 2 epochs. Early stopping...
load model..  da
model loaded: da
Info: biLSTM
        model: da
        in_dim: 64
        h_dim: 100
        c_in_dim: 100
        c_h_dim: 100
        h_layers: 1
        embeds: None
        crf: False
        viterbi_loss: False
        transition_matrix: None
        builder: lstmc
        mlp: 0
        ac_mlp: tanh
        ac: tanh
        raw: False
        dictionary: None
        type_constraint: False
        embed_lex: False
        lex_dim: 0
        output: None
        output_confidences: False
        save_embeds: None
        save_lexembeds: None
        save_cwembeds: None
        save_lwembeds: None
        mimickx_model: None
        iters: 50
        sigma: 0.2
        trainer: sgd
        learning_rate: 0
        patience: 2
        log_losses: False
        word_dropout_rate: 0.25
        char_dropout_rate: 0.0
        disable_backprob_embeds: True
        initializer: constant
        seed: None
        dynet_mem: 1500
        dynet_gpus: 0
        dynet_autobatch: 0
        minibatch_size: 1
load model..  da
model loaded: da


# Training with CRF

python src/structbilty.py --dynet-mem 1500 --train data/da-ud-train.conllu --crf --test data/da-ud-test.conllu --iters 10 --model crf-da

Testing task0
*******


Task0 test accuracy on 1 items: 0.9611
Done. Took 2104.67 seconds in total (testing took 4.31 seconds).
>>> using global decoding (CRF, neg-log loss) <<<
>>> using seed: 4077670 <<<
init parameters
Caching top 5265 words
17553 w features, 100 c features
build graph
CRF
train..
The dy.parameter(...) call is now DEPRECATED.
        There is no longer need to explicitly add parameters to the computation graph.
        Any used parameter will be added automatically.
iter 0   total loss: 0.71
iter 1   total loss: 0.31
iter 2   total loss: 0.24
iter 3   total loss: 0.19
iter 4   total loss: 0.16
iter 5   total loss: 0.14
iter 6   total loss: 0.12
iter 7   total loss: 0.11
iter 8   total loss: 0.10
iter 9   total loss: 0.08
model stored: da.model
load model..  da
CRF
model loaded: da
Info: biLSTM
        model: da
        in_dim: 64
        h_dim: 100
        c_in_dim: 100
        c_h_dim: 100
        h_layers: 1
        embeds: None
        crf: True
        viterbi_loss: False
        transition_matrix: None
        builder: lstmc
        mlp: 0
        ac_mlp: tanh
        ac: tanh
        raw: False
        dictionary: None
        type_constraint: False
        embed_lex: False
        lex_dim: 0
        output: None
        output_confidences: False
        save_embeds: None
        save_lexembeds: None
        save_cwembeds: None
        save_lwembeds: None
        mimickx_model: None
        iters: 10
        sigma: 0.2
        trainer: sgd
        learning_rate: 0
        patience: 0
        log_losses: False
        word_dropout_rate: 0.25
        char_dropout_rate: 0.0
        disable_backprob_embeds: True
        initializer: constant
        seed: None
        dynet_mem: 1500
        dynet_gpus: 0
        dynet_autobatch: 0
        minibatch_size: 1
load model..  da
CRF
model loaded: da


# Testing 
python src/structbilty.py --model da --test data/da-ud-test.conllu --output predictions/test-da.out

Testing task0
*******


Task0 test accuracy on 1 items: 0.9585
Done. Took 4.77 seconds in total (testing took 3.27 seconds).
>>> using seed: 80516342 <<<
load model..  da
model loaded: da
Info: biLSTM
        model: da
        in_dim: 64
        h_dim: 100
        c_in_dim: 100
        c_h_dim: 100
        h_layers: 1
        embeds: None
        crf: False
        viterbi_loss: False
        transition_matrix: None
        builder: lstmc
        mlp: 0
        ac_mlp: tanh
        raw: False
        dictionary: None
        type_constraint: False
        embed_lex: False
        lex_dim: 0
        output: predictions/test-da.out
        output_confidences: False
        save_embeds: None
        save_lexembeds: None
        save_cwembeds: None
        save_lwembeds: None
        mimickx_model: None
        learning_rate: 0
        log_losses: False
        char_dropout_rate: 0.0
        disable_backprob_embeds: True
        seed: None
        dynet_gpus: 0
        dynet_autobatch: 0
        minibatch_size: 1
load model..  da
model loaded: da



# Testing crf
python src/structbilty.py --model crf-da --test data/da-ud-test.conllu --output predictions/test-crf-da.out

Testing task0
*******


Task0 test accuracy on 1 items: 0.9611
Done. Took 6.14 seconds in total (testing took 4.49 seconds).
>>> using seed: 82533641 <<<
load model..  da
CRF
model loaded: da
Info: biLSTM
        model: da
        in_dim: 64
        h_dim: 100
        c_in_dim: 100
        c_h_dim: 100
        h_layers: 1
        embeds: None
        crf: False
        viterbi_loss: False
        transition_matrix: None
        builder: lstmc
        mlp: 0
        ac_mlp: tanh
        raw: False
        dictionary: None
        type_constraint: False
        embed_lex: False
        lex_dim: 0
        output: predictions/test-da.out
        output_confidences: False
        save_embeds: None
        save_lexembeds: None
        save_cwembeds: None
        save_lwembeds: None
        mimickx_model: None
        learning_rate: 0
        log_losses: False
        char_dropout_rate: 0.0
        disable_backprob_embeds: True
        seed: None
        dynet_gpus: 0
        dynet_autobatch: 0
        minibatch_size: 1
load model..  da
CRF
model loaded: da
