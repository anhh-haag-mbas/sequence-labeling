import tensorflow as tf
from tensorflow.contrib.rnn import LSTMCell as LstmCell
from tensorflow.nn import embedding_lookup
from tensorflow.layers import Layer

class Embedding(Layer):
  def __init__(self, vocab_size, embedding_size, **kwargs):
    super(Embedding, self).__init__(**kwargs)

    # The amount of words in our vocabulary
    self.vocab_size = vocab_size
    # When embedding a how, how many dimensions do the embedded vector have
    self.embedding_size = embedding_size

  def build(self, input_shape):
    self.embedding = self.add_variable("embedding", shape=[self.vocab_size, self.embedding_size])

  def call(self, input):
    return embedding_lookup(self.embedding, input)

class Sentences:
  def __init__(self, conll_file_path):
    self.path = conll_file_path
    self.unknown_id = 0
    self.sentence_length = self.calculate_sentence_length()
    self.training_word_ids, self.id_by_word = self.construct_training_word_ids()
    self.training_tag_ids, self.id_by_tag = self.construct_training_tag_ids()
    self.validation_word_ids = self.construct_validation_word_ids()
    self.validation_tag_ids = self.construct_validation_tag_ids()
    self.test_word_ids = self.construct_test_word_ids()
    self.test_tag_ids = self.construct_test_tag_ids()
    self.word_count = len(self.id_by_word)
    self.tag_count = len(self.id_by_tag)

  def construct_training_word_ids(self):
    return self.training_groups_to_ids(self.training_words())

  def construct_training_tag_ids(self):
    return self.training_groups_to_ids(self.training_tags())

  def construct_validation_word_ids(self):
    return self.groups_to_ids(self.validation_words(), self.id_by_word)

  def construct_validation_tag_ids(self):
    return self.groups_to_ids(self.validation_tags(), self.id_by_tag)

  def construct_test_word_ids(self):
    return self.groups_to_ids(self.test_words(), self.id_by_word)

  def construct_test_tag_ids(self):
    return self.groups_to_ids(self.test_tags(), self.id_by_tag)

  def training_groups_to_ids(self, groups):
    id_by_item = {"__unk__": self.unknown_id}
    id_groups = []
    id_group = []
    for group in groups:
      for item in group:
        if item not in id_by_item:
          id_by_item[item] = len(id_by_item)
        id_group += [id_by_item[item]]
      id_groups += [id_group]
      id_group = []
    return [id_groups, id_by_item]

  def groups_to_ids(self, groups, id_by_item):
    id_groups = []
    id_group = []
    for group in groups:
      for item in group:
        if item in id_by_item:
          id_group += [id_by_item[item]]
        else:
          id_group += [self.unknown_id]
      id_groups += [id_group]
      id_group = []
    return id_groups


  def training_words(self):
    return self.padded_words()[0:900]

  def training_tags(self):
    return self.padded_tags()[0:900]

  def validation_words(self):
    return self.padded_words()[900:950]

  def validation_tags(self):
    return self.padded_tags()[900:950]

  def test_words(self):
    return self.padded_words()[950:1000]

  def test_tags(self):
    return self.padded_tags()[950:1000]

  def calculate_sentence_length(self):
    longest_sentence = max(self.items(), key=len)
    return len(longest_sentence)

  def padded_words(self):
    return [self.pad(group, "__pad__", self.sentence_length) for group in self.words()]

  def padded_tags(self):
    return [self.pad(group, "__pad__", self.sentence_length) for group in self.tags()]

  def pad(self, items, padding, size):
    while(len(items) < size):
      items += [padding]
    return items

  def words(self):
    return [[items[1] for items in group] for group in self.items()]

  def tags(self):
    return [[items[3] for items in group] for group in self.items()]

  def items(self):
    return [self.items_from_lines(group) for group in self.groups()]

  def items_from_lines(self, lines):
    return [line.split("\t") for line in lines if not self.should_skip(line)]

  def should_skip(self, line):
    if line.strip() == "": return True
    if line.startswith("#"): return True
    return False

  def groups(self):
    groups = []
    group = []

    for line in self.lines():
      group += [line]
      if line.strip() == "":
        groups += [group]
        group = []

    return groups

  def lines(self):
    with open(self.path, "r") as file:
      for line in file: yield line
