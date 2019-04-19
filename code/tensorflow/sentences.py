import numpy as np

def reverse_dict(dict):
    return {v: k for k, v in dict.items()}

class Sentences:
    def __init__(self, task, language_code, id_by_word, id_by_tag):
        self.task = task
        self.language_code = language_code

        self.id_by_word = id_by_word
        self.word_by_id = reverse_dict(self.id_by_word.items())
        self.word_count = len(self.id_by_tag)

        self.id_by_tag = id_by_tag
        self.tag_by_id = reverse_dict(self.id_by_tag.items())
        self.tag_count = len(self.id_by_tag)

        self.word_unknown_id = self.id_by_word["<UNK>"]
        self.word_padding_id = self.id_by_word["<PAD>"]

        self.tag_unknown_id = self.id_by_tag["<UNK>"]
        self.tag_padding_id = self.id_by_tag["<PAD>"]


class Sentence:
    def __init__(self, words, tags):
        assert len(words) == len(tags)
        self.words = words
        self.tags = tags
        self.length = len(self.words)

    def pad(self, minimum_length, pad_word, pad_tag):
        s = self.copy()
        missing_items = minimum_length - s.length
        words = s.words + [pad_word] * missing_items
        tags = s.tags = [pad_tag] * missing_items
        return Sentence(words, tags)

    def copy(self):
        return Sentence(self.words, self.tags)

    @staticmethod
    def from_ner_lines(lines):
        words = [line.split("\t")[0].strip() for line in lines]
        tags = [line.split("\t")[1].strip() for line in lines]
        return Sentence(words, tags)

    @staticmethod
    def from_pos_lines(lines):
        words = [line.split("\t")[1].strip() for line in lines]
        tags = [line.split("\t")[3].strip() for line in lines]
        return Sentence(words, tags)


class Sentences:
    def __init__(self, task, language_code, id_by_word):
        """
        :param task: "pos", "ner"
        :param language_code: e.g. "da", "en", "ch", etc.
        """
        self.task = task
        self.language_code = language_code
        # TODO: Padding/unknown words are different in polyglot/fasttext
        self.sentence_length = self.calculate_sentence_length(
            self.items(self.groups(self.lines(self.pos_file_path(self.task, self.language_code, "training")))))

        self.id_by_word = id_by_word
        self.training_word_ids, self.id_by_word = self.construct_training_word_ids()
        self.word_by_id = {v: k for k, v in self.id_by_word.items()}
        self.training_tag_ids, self.id_by_tag = self.construct_training_tag_ids()
        self.tag_by_id = {v: k for k, v in self.id_by_tag.items()}

        self.training_lengths = self.lengths(self.unpadded_tags_for("training"))
        self.validation_word_ids = self.construct_validation_word_ids()
        self.validation_tag_ids = self.construct_validation_tag_ids()
        self.validation_lengths = self.lengths(self.unpadded_tags_for("validation"))
        self.testing_word_ids = self.construct_testing_word_ids()
        self.testing_tag_ids = self.construct_testing_tag_ids()
        self.testing_lengths = self.lengths(self.unpadded_tags_for("testing"))
        self.word_count = len(self.id_by_word)
        self.tag_count = len(self.id_by_tag)

        self.word_unknown_id = self.id_by_word["<UNK>"]
        self.word_padding_id = self.id_by_word["<PAD>"]

        self.tag_unknown_id = self.id_by_tag["<UNK>"]
        self.tag_padding_id = self.id_by_tag["<PAD>"]

        self.convert_to_numpy_arrays()

    def convert_to_numpy_arrays(self):
        self.training_word_ids = np.asarray(self.training_word_ids)
        self.training_tag_ids = np.asarray(self.training_tag_ids)
        self.training_lengths = np.asarray(self.training_lengths)
        self.validation_word_ids = np.asarray(self.validation_word_ids)
        self.validation_tag_ids = np.asarray(self.validation_tag_ids)
        self.validation_lengths = np.asarray(self.validation_lengths)
        self.testing_word_ids = np.asarray(self.testing_word_ids)
        self.testing_tag_ids = np.asarray(self.testing_tag_ids)
        self.testing_lengths = np.asarray(self.testing_lengths)

    def lengths(self, sentences):
        return [len(sentence) for sentence in sentences]

    def construct_training_word_ids(self):
        return self.groups_to_ids(self.training_words(), self.id_by_word), self.id_by_word

    def construct_training_tag_ids(self):
        return self.training_groups_to_ids(self.training_tags())

    def construct_validation_word_ids(self):
        return self.groups_to_ids(self.validation_words(), self.id_by_word)

    def construct_validation_tag_ids(self):
        return self.groups_to_ids(self.validation_tags(), self.id_by_tag)

    def construct_testing_word_ids(self):
        return self.groups_to_ids(self.testing_words(), self.id_by_word)

    def construct_testing_tag_ids(self):
        return self.groups_to_ids(self.testing_tags(), self.id_by_tag)

    def training_groups_to_ids(self, groups):
        id_by_item = {"<PAD>": 0, "<UNK>": 1}
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
                    id_group += [id_by_item["<UNK>"]]
            id_groups += [id_group]
            id_group = []
        return id_groups

    def training_words(self):
        return self.words_for("training")

    def training_tags(self):
        return self.tags_for("training")

    def validation_words(self):
        return self.words_for("validation")

    def validation_tags(self):
        return self.tags_for("validation")

    def testing_words(self):
        return self.words_for("testing")

    def testing_tags(self):
        return self.tags_for("testing")

    def words_for(self, data_type):
        return self.padded_words(self.words(
            self.items(self.groups(self.lines(self.pos_file_path(self.task, self.language_code, data_type))))))

    def unpadded_words_for(self, data_type):
        return self.words(
            self.items(self.groups(self.lines(self.pos_file_path(self.task, self.language_code, data_type)))))

    def tags_for(self, data_type):
        return self.padded_tags(self.tags(
            self.items(self.groups(self.lines(self.pos_file_path(self.task, self.language_code, data_type))))))

    def unpadded_tags_for(self, data_type):
        return self.tags(
            self.items(self.groups(self.lines(self.pos_file_path(self.task, self.language_code, data_type)))))

    def pos_file_path(self, language_code, data_type):
        return f"../../data/pos/{language_code}/{data_type}.conllu"

    def ner_file_path(self, language_code, data_type):
        return f"../../data/ner/{language_code}/{data_type}.bio"

    def calculate_sentence_length(self, items):
        longest_sentence = max(items, key=len)
        return len(longest_sentence)

    def padded_words(self, words):
        return [self.pad(group, "<PAD>", self.sentence_length) for group in words]

    def padded_tags(self, tags):
        return [self.pad(group, "<PAD>", self.sentence_length) for group in tags]

    def pad(self, items, padding, size):
        while len(items) < size:
            items += [padding]
        return items

    def words(self, items):
        return [[items[1] for items in group] for group in items]

    def tags(self, items):
        return [[items[3] for items in group] for group in items]

    def items(self, groups):
        return [self.items_from_lines(group) for group in groups]

    def items_from_lines(self, lines):
        return [line.split("\t") for line in lines if not self.should_skip(line)]

    def should_skip(self, line):
        if line.strip() == "" or line.startswith("#"):
            return True
        return False

    def groups(self, lines):
        groups = []
        group = []

        for line in lines:
            group += [line]
            if line.strip() == "":
                groups += [group]
                group = []

        return groups

    def lines(self, path):
        with open(path, "r") as file:
            for line in file:
                yield line
