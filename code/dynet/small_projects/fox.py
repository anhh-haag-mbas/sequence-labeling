import sys
import random

import dynet as dy
import ipdb

from collections import defaultdict
from itertools import count

LAYERS = 1
INPUT_DIM = 50
HIDDEN_DIM = 50

characters = list("abcdefghijklmnopqrstuvwxyz ")
characters.append("<EOS>")

int2char = list(characters)
char2int = { c:i for i, c in enumerate(characters) }

VOCAB_SIZE = len(characters)

pc = dy.ParameterCollection()

srnn = dy.SimpleRNNBuilder(LAYERS, INPUT_DIM, HIDDEN_DIM, pc)
lstm = dy.LSTMBuilder(LAYERS, INPUT_DIM, HIDDEN_DIM, pc)

# add parameters for the hidden->output part for both lstm and srnn
params_lstm = {}
params_srnn = {}

for params in [params_lstm, params_srnn]:
    params["lookup"] = pc.add_lookup_parameters((VOCAB_SIZE, INPUT_DIM))
    params["R"] = pc.add_parameters((VOCAB_SIZE, HIDDEN_DIM))
    params["bias"] = pc.add_parameters((VOCAB_SIZE))

# return compute loss of RNN for one sentence
def do_one_sentence(rnn, params, sentence):
# setup the sentence
    dy.renew_cg()
    s0 = rnn.initial_state()
            
    R = params["R"]
    bias = params["bias"]
    lookup = params["lookup"]
    sentence = ["<EOS>"] + list(sentence) + ["<EOS>"]
    sentence = [char2int[c] for c in sentence]
    s = s0
    loss = []
    for char,next_char in zip(sentence,sentence[1:]):
        s = s.add_input(lookup[char])
        probs = dy.softmax(R*s.output() + bias)
        loss.append( -dy.log(dy.pick(probs,next_char)) )
    loss = dy.esum(loss)
    return loss


# generate from model:
def generate(rnn, params):
    def sample(probs):
        rnd = random.random()
        for i,p in enumerate(probs):
            rnd -= p
            if rnd <= 0: break
        return i

    # setup the sentence
    dy.renew_cg()
    s0 = rnn.initial_state()

    R = params["R"]
    bias = params["bias"]
    lookup = params["lookup"]

    s = s0.add_input(lookup[char2int["<EOS>"]])
    out=[]

    while True:
        probs = dy.softmax(R*s.output() + bias)
        probs = probs.vec_value()
        next_char = sample(probs)
        out.append(int2char[next_char])
        if out[-1] == "<EOS>": break
        s = s.add_input(lookup[next_char])
    return "".join(out[:-1]) # strip the <EOS>


# train, and generate every 5 samples
def train(rnn, params, sentence):
    trainer = dy.SimpleSGDTrainer(pc)
    for i in range(200):
        loss = do_one_sentence(rnn, params, sentence)
        loss_value = loss.value()
        loss.backward()
        trainer.update()
        if i % 5 == 0: 
            print("%.10f" % loss_value, end="\t")
            print(generate(rnn, params))

ipdb.set_trace()
