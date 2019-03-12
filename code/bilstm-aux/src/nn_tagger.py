import array
import random
import sys
import numpy as np
import dill
import _dynet as dynet

from collections import Counter, defaultdict
from lib.mnnl import FFSequencePredictor, Layer, BiRNNSequencePredictor, CRFSequencePredictor, is_in_dict
from lib.mio import load_embeddings_file, load_dict
from lib.constants import UNK, WORD_START, WORD_END, START_TAG, END_TAG
from lib.mmappers import TRAINER_MAP, ACTIVATION_MAP, INITIALIZER_MAP, BUILDERS

class NNTagger(object):

    # turn dynamic allocation off by defining slots
    __slots__ = ['w2i', 'c2i', 'wcount', 'ccount','wtotal','ctotal','w2c_cache','w_dropout_rate','c_dropout_rate',
                  'task2tag2idx', 'model', 'in_dim', 'c_in_dim', 'c_h_dim','h_dim', 'activation',
                 'noise_sigma', 'pred_layer', 'mlp', 'activation_mlp', 'backprob_embeds', 'initializer',
                 'h_layers', 'predictors', 'wembeds', 'cembeds', 'embeds_file', 'char_rnn', 'trainer',
                 'builder', 'crf', 'viterbi_loss', 'mimickx_model_path', 'mimickx_model',
                 'dictionary',  'dictionary_values', 'path_to_dictionary', 'lex_dim', 'type_constraint',
                 'embed_lex', 'l2i', 'lembeds']

    def __init__(self,in_dim,h_dim,c_in_dim,c_h_dim,h_layers,pred_layer,learning_algo="sgd", learning_rate=0,
                 embeds_file=None,activation=ACTIVATION_MAP["tanh"],mlp=0,activation_mlp=ACTIVATION_MAP["rectify"],
                 backprob_embeds=True,noise_sigma=0.1, w_dropout_rate=0.25, c_dropout_rate=0.25,
                 initializer=INITIALIZER_MAP["glorot"], builder=BUILDERS["lstmc"], crf=False, viterbi_loss=False,
                 mimickx_model_path=None, dictionary=None, type_constraint=False,
                 lex_dim=0, embed_lex=False):
        self.w2i = {}  # word to index mapping
        self.c2i = {}  # char to index mapping
        self.w2c_cache = {} # word to char index cache for frequent words
        self.wcount = None # word count
        self.ccount = None # char count
        self.task2tag2idx = {} # need one dictionary per task
        self.pred_layer = [int(layer) for layer in pred_layer] # at which layer to predict each task
        self.model = dynet.ParameterCollection() #init model
        self.in_dim = in_dim
        self.h_dim = h_dim
        self.c_in_dim = c_in_dim
        self.c_h_dim = c_h_dim
        self.w_dropout_rate = w_dropout_rate
        self.c_dropout_rate = c_dropout_rate
        self.activation = activation
        self.mlp = mlp
        self.activation_mlp = activation_mlp
        self.noise_sigma = noise_sigma
        self.h_layers = h_layers
        self.predictors = {"inner": [], "output_layers_dict": {}, "task_expected_at": {} } # the inner layers and predictors
        self.wembeds = None # lookup: embeddings for words
        self.cembeds = None # lookup: embeddings for characters
        self.lembeds = None # lookup: embeddings for lexical features (optional)
        self.embeds_file = embeds_file
        trainer_algo = TRAINER_MAP[learning_algo]
        if learning_rate > 0:
            ### TODO: better handling of additional learning-specific parameters
            self.trainer = trainer_algo(self.model, learning_rate=learning_rate)
        else:
            # using default learning rate
            self.trainer = trainer_algo(self.model)
        self.backprob_embeds = backprob_embeds
        self.initializer = initializer
        self.char_rnn = None # biRNN for character input
        self.builder = builder # default biRNN is an LSTM
        self.crf = crf
        self.viterbi_loss = viterbi_loss
        self.mimickx_model_path = mimickx_model_path
        if mimickx_model_path: # load
            self.mimickx_model = load_model(mimickx_model_path)
        self.dictionary = None
        self.type_constraint = type_constraint
        self.embed_lex = False
        self.l2i = {UNK: 0}  # lex feature to index mapping
        if dictionary:
            self.dictionary, self.dictionary_values = load_dict(dictionary)
            self.path_to_dictionary = dictionary
            if type_constraint:
                self.lex_dim = 0
            else:
                if embed_lex:
                    self.lex_dim = lex_dim
                    self.embed_lex = True
                    print("Embed lexical features")
                    # register property indices
                    for prop in self.dictionary_values:
                        self.l2i[prop] = len(self.l2i)
                else:
                    self.lex_dim = len(self.dictionary_values) #n-hot encoding
                print("Lex_dim: {}".format(self.lex_dim), file=sys.stderr)
        else:
            self.dictionary = None
            self.path_to_dictionary = None
            self.lex_dim = 0

    def fit(self, train, num_iterations, dev=None, model_path=None, patience=0, minibatch_size=0, log_losses=False):
        """
        train the tagger
        """
        losses_log = {} # log losses

        print("init parameters")
        self.init_parameters(train)

        # init lookup parameters and define graph
        print("build graph")
        self.build_computation_graph(len(self.w2i),  len(self.c2i))

        update_embeds = True
        if self.backprob_embeds == False: ## disable backprob into embeds
            print(">>> disable wembeds update <<<")
            update_embeds = False
            
        best_val_acc, epochs_no_improvement = 0.0, 0

        if dev and model_path is not None and patience > 0:
            print('Using early stopping with patience of {}...'.format(patience))

        batch = []
        print("train..")
        for iteration in range(num_iterations):

            total_loss=0.0
            total_tagged=0.0

            indices = [i for i in range(len(train.seqs))]
            random.shuffle(indices)

            loss_accum_loss = defaultdict(float)
            loss_accum_tagged = defaultdict(float)

            for idx in indices:
                seq = train.seqs[idx]

                if seq.task_id not in losses_log:
                    losses_log[seq.task_id] = [] #initialize

                if minibatch_size > 1:
                    # accumulate instances for minibatch update
                    loss1 = self.predict(seq, train=True, update_embeds=update_embeds)
                    total_tagged += len(seq.words)
                    batch.append(loss1)
                    if len(batch) == minibatch_size:
                        loss = dynet.esum(batch)
                        total_loss += loss.value()

                        # logging
                        loss_accum_tagged[seq.task_id] += len(seq.words)
                        loss_accum_loss[seq.task_id] += loss.value()

                        loss.backward()
                        self.trainer.update()
                        dynet.renew_cg()  # use new computational graph for each BATCH when batching is active
                        batch = []
                else:
                    dynet.renew_cg() # new graph per item
                    loss1 = self.predict(seq, train=True, update_embeds=update_embeds)
                    total_tagged += len(seq.words)
                    lv = loss1.value()
                    total_loss += lv

                    # logging
                    loss_accum_tagged[seq.task_id] += len(seq.words)
                    loss_accum_loss[seq.task_id] += loss1.value()

                    loss1.backward()
                    self.trainer.update()

            print("iter {2} {0:>12}: {1:.2f}".format("total loss", total_loss/total_tagged, iteration))

            # log losses
            for task_id in sorted(losses_log):
                losses_log[task_id].append(loss_accum_loss[task_id] / loss_accum_tagged[task_id])

            if log_losses:
                dill.dump(losses_log, open(model_path + ".model" + ".losses.pickle", "wb"))

            if dev:
                # evaluate after every epoch
                correct, total = self.evaluate(dev, "task0")
                val_accuracy = correct/total
                print("dev accuracy: {0:.4f}".format(val_accuracy))

                if val_accuracy > best_val_acc:
                    print('Accuracy {0:.4f} is better than best val accuracy '
                          '{1:.4f}.'.format(val_accuracy, best_val_acc))
                    best_val_acc = val_accuracy
                    epochs_no_improvement = 0
                    save(self, model_path)
                else:
                    print('Accuracy {0:.4f} is worse than best val loss {1:.4f}.'.format(val_accuracy, best_val_acc))
                    epochs_no_improvement += 1

                if patience > 0:
                    if epochs_no_improvement == patience:
                        print('No improvement for {} epochs. Early stopping...'.format(epochs_no_improvement))
                        break

    def set_indices(self, w2i, c2i, task2t2i, w2c_cache, l2i=None):
        """ helper function for loading model"""
        for task_id in task2t2i:
            self.task2tag2idx[task_id] = task2t2i[task_id]
        self.w2i = w2i
        self.c2i = c2i
        self.w2c_cache = w2c_cache
        self.l2i = l2i

    def set_counts(self, wcount, wtotal, ccount, ctotal):
        """ helper function for loading model"""
        self.wcount = wcount
        self.wtotal = wtotal
        self.ccount = ccount
        self.ctotal = ctotal

    def build_computation_graph(self, num_words, num_chars):
        """
        build graph and link to parameters
        self.predictors, self.char_rnn, self.wembeds, self.cembeds =
        """
        ## initialize word embeddings
        if self.embeds_file:
            print("loading embeddings")
            embeddings, emb_dim = load_embeddings_file(self.embeds_file)
            assert(emb_dim==self.in_dim)
            num_words=len(set(embeddings.keys()).union(set(self.w2i.keys()))) # initialize all with embeddings
            # init model parameters and initialize them
            self.wembeds = self.model.add_lookup_parameters((num_words, self.in_dim), init=self.initializer)

            init=0
            for word in embeddings.keys():
                if word not in self.w2i:
                    self.w2i[word]=len(self.w2i.keys()) # add new word
                    self.wembeds.init_row(self.w2i[word], embeddings[word])
                    init +=1
                elif word in embeddings:
                    self.wembeds.init_row(self.w2i[word], embeddings[word])
                    init += 1
            print("initialized: {}".format(init))
            del embeddings # clean up
        else:
            self.wembeds = self.model.add_lookup_parameters((num_words, self.in_dim), init=self.initializer)

        ## initialize character embeddings
        self.cembeds = None
        if self.c_in_dim > 0:
            self.cembeds = self.model.add_lookup_parameters((num_chars, self.c_in_dim), init=self.initializer)
        if self.lex_dim > 0 and self.embed_lex:
            # +1 for UNK property
            self.lembeds = self.model.add_lookup_parameters((len(self.dictionary_values)+1, self.lex_dim), init=dynet.GlorotInitializer()) #init=self.initializer)

        # make it more flexible to add number of layers as specified by parameter
        layers = [] # inner layers
        output_layers_dict = {}   # from task_id to actual softmax predictor
        for layer_num in range(0,self.h_layers):
            if layer_num == 0:
                if self.c_in_dim > 0:
                    # in_dim: size of each layer
                    if self.lex_dim > 0 and self.embed_lex:
                        lex_embed_size = self.lex_dim * len(self.dictionary_values)
                        f_builder = self.builder(1, self.in_dim+self.c_h_dim*2+lex_embed_size, self.h_dim, self.model)
                        b_builder = self.builder(1, self.in_dim+self.c_h_dim*2+lex_embed_size, self.h_dim, self.model)
                    else:
                        f_builder = self.builder(1, self.in_dim + self.c_h_dim * 2 + self.lex_dim, self.h_dim, self.model)
                        b_builder = self.builder(1, self.in_dim + self.c_h_dim * 2 + self.lex_dim, self.h_dim, self.model)
                else:
                    f_builder = self.builder(1, self.in_dim+self.lex_dim, self.h_dim, self.model)
                    b_builder = self.builder(1, self.in_dim+self.lex_dim, self.h_dim, self.model)

                layers.append(BiRNNSequencePredictor(f_builder, b_builder)) #returns forward and backward sequence
            else:
                # add inner layers (if h_layers >1)
                f_builder = self.builder(1, self.h_dim, self.h_dim, self.model)
                b_builder = self.builder(1, self.h_dim, self.h_dim, self.model)
                layers.append(BiRNNSequencePredictor(f_builder, b_builder))

        # store at which layer to predict task
        task2layer = {task_id: out_layer for task_id, out_layer in zip(self.task2tag2idx, self.pred_layer)}
        if len(task2layer) > 1:
            print("task2layer", task2layer)
        for task_id in task2layer:
            task_num_labels= len(self.task2tag2idx[task_id])
            if not self.crf:
                output_layers_dict[task_id] = FFSequencePredictor(self.task2tag2idx[task_id], Layer(self.model, self.h_dim*2, task_num_labels,
                                                                                                    dynet.softmax, mlp=self.mlp, mlp_activation=self.activation_mlp))
            else:
                print("CRF")
                output_layers_dict[task_id] = CRFSequencePredictor(self.model, task_num_labels,
                                                                   self.task2tag2idx[task_id],
                                                                   Layer(self.model, self.h_dim * 2, task_num_labels,
                                                                        None, mlp=self.mlp,
                                                                        mlp_activation=self.activation_mlp), viterbi_loss=self.viterbi_loss)

        self.char_rnn = BiRNNSequencePredictor(self.builder(1, self.c_in_dim, self.c_h_dim, self.model),
                                          self.builder(1, self.c_in_dim, self.c_h_dim, self.model))

        self.predictors = {}
        self.predictors["inner"] = layers
        self.predictors["output_layers_dict"] = output_layers_dict
        self.predictors["task_expected_at"] = task2layer

    def _drop(self, x, xcount, dropout_rate):
        """
        drop x if x is less frequent (cf. Kiperwasser & Goldberg, 2016)
        """
        return random.random() > (xcount.get(x)/(dropout_rate+xcount.get(x)))
        
    def get_features(self, words, train=False, update=True):
        """
        get feature representations
        """
        # word embeddings
        wfeatures = np.array([self.get_w_repr(word, train=train, update=update) for word in words])

        lex_features = []
        if self.dictionary and not self.type_constraint:
            ## add lexicon features
            lex_features = np.array([self.get_lex_repr(word) for word in words])
        # char embeddings
        if self.c_in_dim > 0:
            cfeatures = [self.get_c_repr(word, train=train) for word in words]
            if len(lex_features) > 0:
                lex_features = dynet.inputTensor(lex_features)
                features = [dynet.concatenate([w,c,l]) for w,c,l in zip(wfeatures,cfeatures,lex_features)]
            else:
                features = [dynet.concatenate([w, c]) for w, c in zip(wfeatures, cfeatures)]
        else:
            features = wfeatures
        if train: # only do at training time
            features = [dynet.noise(fe,self.noise_sigma) for fe in features]
        return features

    def predict(self, seq, train=False, output_confidences=False, unk_tag=None, update_embeds=True):
        """
        predict tags for a sentence represented as char+word embeddings and compute losses for this instance
        """
        if not train:
            dynet.renew_cg()
        features = self.get_features(seq.words, train=train, update=update_embeds)

        output_expected_at_layer = self.predictors["task_expected_at"][seq.task_id]
        output_expected_at_layer -=1

        # go through layers
        # input is now combination of w + char emb
        prev = features
        prev_rev = features
        num_layers = self.h_layers

        for i in range(0,num_layers):
            predictor = self.predictors["inner"][i]
            forward_sequence, backward_sequence = predictor.predict_sequence(prev, prev_rev)        
            if i > 0 and self.activation:
                # activation between LSTM layers
                forward_sequence = [self.activation(s) for s in forward_sequence]
                backward_sequence = [self.activation(s) for s in backward_sequence]

            if i == output_expected_at_layer:
                output_predictor = self.predictors["output_layers_dict"][seq.task_id]
                concat_layer = [dynet.concatenate([f, b]) for f, b in zip(forward_sequence,reversed(backward_sequence))]

                if train and self.noise_sigma > 0.0:
                    concat_layer = [dynet.noise(fe,self.noise_sigma) for fe in concat_layer]
                # fill-in predictions and get loss per tag
                losses = output_predictor.predict_sequence(seq, concat_layer,
                                                           train=train, output_confidences=output_confidences,
                                                           unk_tag=unk_tag, dictionary=self.dictionary,
                                                           type_constraint=self.type_constraint)

            prev = forward_sequence
            prev_rev = backward_sequence 

        if train:
            # return losses
            return losses
        else:
            return seq.pred_tags, seq.tag_confidences

    def output_preds(self, seq, raw=False, output_confidences=False):
        """
        output predictions to a file
        """
        i = 0
        for w, g, p in zip(seq.words, seq.tags, seq.pred_tags):
            if raw:
                if output_confidences:
                    print(u"{0}\t{1}\t{2:.2f}".format(w, p, seq.tag_confidences[i]))
                else:
                    print(u"{}\t{}".format(w, p))  # do not print DUMMY tag when --raw is on
            else:
                if output_confidences:
                    print(u"{0}\t{1}\t{2}\t{3:.2f}".format(w, g, p, seq.tag_confidences[i]))
                else:
                    print(u"{}\t{}\t{}".format(w, g, p))
            i += 1
        print("")

    def evaluate(self, test_file, task_id, output_predictions=None, raw=False, output_confidences=False, unk_tag=None):
        """
        compute accuracy on a test file, optionally output to file
        """
        correct = 0
        total = 0

        for seq in test_file:
            if seq.task_id != task_id:
                continue # we evaluate only on a specific task
            self.predict(seq, output_confidences=output_confidences, unk_tag=unk_tag)
            if output_predictions:
                self.output_preds(seq, raw=raw, output_confidences=output_confidences)
            correct_inst, total_inst = seq.evaluate()
            correct+=correct_inst
            total+= total_inst
        return correct, total

    def get_w_repr(self, word, train=False, update=True):
        """
        Get representation of word (word embedding)
        """
        if train:
            if self.w_dropout_rate > 0.0:
                w_id = self.w2i[UNK] if self._drop(word, self.wcount, self.w_dropout_rate) else self.w2i.get(word, self.w2i[UNK])
        else:
            if self.mimickx_model_path: # if given use MIMICKX
                if word not in self.w2i: #
                    #print("predict with MIMICKX for: ", word)
                    return dynet.inputVector(self.mimickx_model.predict(word).npvalue())
            w_id = self.w2i.get(word, self.w2i[UNK])
        if not update:
            return dynet.nobackprop(self.wembeds[w_id])
        else:
            return self.wembeds[w_id] 

    def get_c_repr(self, word, train=False):
        """
        Get representation of word via characters sub-LSTMs
        """
        # get representation for words
        if word in self.w2c_cache:
            chars_of_token = self.w2c_cache[word]
            if train:
                chars_of_token = [self._drop(c, self.ccount, self.c_dropout_rate) for c in chars_of_token]
        else:
            chars_of_token = array.array('I',[self.c2i[WORD_START]]) + array.array('I',[self.get_c_idx(c, train=train) for c in word]) + array.array('I',[self.c2i[WORD_END]])

        char_feats = [self.cembeds[c_id] for c_id in chars_of_token]
        # use last state as word representation
        f_char, b_char = self.char_rnn.predict_sequence(char_feats, char_feats)
        return dynet.concatenate([f_char[-1], b_char[-1]])

    def get_c_idx(self, c, train=False):
        """ helper function to get index of character"""
        if self.c_dropout_rate > 0.0 and train and self._drop(c, self.ccount, self.c_dropout_rate):
            return self.c2i.get(UNK)
        else:
            return self.c2i.get(c, self.c2i[UNK])

    def get_lex_repr(self, word):
        """
        Get representation for lexical feature
        """
        if not self.embed_lex: ## n-hot representation
            n_hot = np.zeros(len(self.dictionary_values))
            values = is_in_dict(word, self.dictionary)
            if values:
                for v in values:
                    n_hot[self.dictionary_values.index(v)] = 1.0
            return n_hot
        else:
            lex_feats = []
            for property in self.dictionary_values:
                values = is_in_dict(word, self.dictionary)
                if values:
                    if property in values:
                        lex_feats.append(self.lembeds[self.l2i[property]].npvalue())
                    else:
                        lex_feats.append(self.lembeds[self.l2i[UNK]].npvalue())
                else:
                    lex_feats.append(self.lembeds[self.l2i[UNK]].npvalue()) # unknown word
            return np.concatenate(lex_feats)

    def init_parameters(self, train_data):
        """init parameters from training data"""
        # word 2 indices and tag 2 indices
        self.w2i = {}  # word to index
        self.c2i = {}  # char to index
        self.task2tag2idx = {}  # id of the task -> tag2idx

        self.w2i[UNK] = 0  # unk word / OOV
        self.c2i[UNK] = 0  # unk char
        self.c2i[WORD_START] = 1  # word start
        self.c2i[WORD_END] = 2  # word end index

        # word and char counters
        self.wcount = Counter()
        self.ccount = Counter()

        for seq in train_data:
            self.wcount.update([w for w in seq.words])
            self.ccount.update([c for w in seq.words for c in w])

            if seq.task_id not in self.task2tag2idx:
                self.task2tag2idx[seq.task_id] = {"<START>": START_TAG, "<END>": END_TAG}

            # record words and chars
            for word, tag in zip(seq.words, seq.tags):
                if word not in self.w2i:
                    self.w2i[word] = len(self.w2i)

                if self.c_in_dim > 0:
                    for char in word:
                        if char not in self.c2i:
                            self.c2i[char] = len(self.c2i)

                if tag not in self.task2tag2idx[seq.task_id]:
                    self.task2tag2idx[seq.task_id][tag] = len(self.task2tag2idx[seq.task_id])

        n = int(len(self.w2i) * 0.3) # top 30%
        print("Caching top {} words".format(n))
        for word in self.wcount.most_common(n):
            self.w2c_cache[word] = array.array('I', [self.c2i[WORD_START]]) + array.array('I', [self.get_c_idx(c) for c in word]) + array.array('I', [self.c2i[WORD_END]])
        # get total counts
        self.wtotal = np.sum([self.wcount[w] for w in self.wcount])
        self.ctotal = np.sum([self.ccount[c] for c in self.ccount])
        print("{} w features, {} c features".format(len(self.w2i), len(self.c2i)))
        #print(self.wtotal, self.ctotal)


    def save_embeds(self, out_filename):
        """
        save final embeddings to file
        :param out_filename: filename
        """
        # construct reverse mapping
        i2w = {self.w2i[w]: w for w in self.w2i.keys()}

        OUT = open(out_filename+".w.emb","w")
        for word_id in i2w.keys():
            wembeds_expression = self.wembeds[word_id]
            word = i2w[word_id]
            OUT.write("{} {}\n".format(word," ".join([str(x) for x in wembeds_expression.npvalue()])))
        OUT.close()


    def save_lex_embeds(self, out_filename):
        """
        save final embeddings to file
        :param out_filename: filename
        """
        # construct reverse mapping
        i2l = {self.l2i[w]: w for w in self.l2i.keys()}

        OUT = open(out_filename+".l.emb","w")
        for lex_id in i2l.keys():
            lembeds_expression = self.lembeds[lex_id]
            lex = i2l[lex_id]
            OUT.write("{} {}\n".format(lex," ".join([str(x) for x in lembeds_expression.npvalue()])))
        OUT.close()


    def save_cw_embeds(self, out_filename):
        """
        save final character-based word-embeddings to file
        :param out_filename: filename
        """
        # construct reverse mapping using word embeddings
        i2cw = {self.w2i[w]: w for w in self.w2i.keys()}

        OUT = open(out_filename+".cw.emb","w")
        for word_id in i2cw.keys():
            word = i2cw[word_id]
            cwembeds = [v.npvalue()[0] for v in self.get_c_repr(word)]
            OUT.write("{} {}\n".format(word," ".join([str(x) for x in cwembeds])))
        OUT.close()


    def save_wordlex_map(self, out_filename):
        """
        save final word-to-lexicon-embedding map to file
        :param out_filename: filename
        """
        # construct reverse mapping using word embeddings
        i2wl = {self.w2i[w]: w for w in self.w2i.keys()}

        OUT = open(out_filename+".wlmap.emb","w")
        for word_id in i2wl.keys():
            word = i2wl[word_id]

            lex_feats = []
            for property in self.dictionary_values:
                values = is_in_dict(word, self.dictionary)
                if values:
                    if property in values:
                        lex_feats.append(property)
                    else:
                        lex_feats.append(UNK)
                else:
                    lex_feats.append(UNK) # unknown word

            OUT.write("{} {}\n".format(word," ".join([str(x) for x in lex_feats])))
        OUT.close()
        
    def save_transition_matrix(self, out_filename):
        """
        save transition matrix
        :param out_filename: filename
        """
        for task_id in self.predictors["output_layers_dict"].keys():
            output_predictor = self.predictors["output_layers_dict"][task_id]
            output_predictor.save_parameters(out_filename)

