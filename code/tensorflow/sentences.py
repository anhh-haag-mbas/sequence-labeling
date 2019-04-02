import numpy as np


class Sentences:
    def __init__(self, task, language_code):
        """
        :param task: "pos", "ner"
        :param language_code: e.g. "da", "en", "ch", etc.
        """
        self.task = task
        self.language_code = language_code
        self.padding_id = 0
        self.unknown_id = 1
        self.sentence_length = self.calculate_sentence_length(self.items(self.groups(self.lines(self.conllu_file_path(self.task, self.language_code, "training")))))
        self.training_word_ids, self.id_by_word = self.construct_training_word_ids()
        self.word_by_id = {v: k for k, v in self.id_by_word.items()}
        self.training_tag_ids, self.id_by_tag = self.construct_training_tag_ids()
        self.tag_by_id = {v: k for k, v in self.id_by_tag.items()}
        self.validation_word_ids = self.construct_validation_word_ids()
        self.validation_tag_ids = self.construct_validation_tag_ids()
        self.testing_word_ids = self.construct_testing_word_ids()
        self.testing_tag_ids = self.construct_testing_tag_ids()
        self.word_count = len(self.id_by_word)
        self.tag_count = len(self.id_by_tag)

        self.convert_to_numpy_arrays()

    def convert_to_numpy_arrays(self):
        self.training_word_ids = np.asarray(self.training_word_ids)
        self.training_tag_ids = np.asarray(self.training_tag_ids)
        self.validation_word_ids = np.asarray(self.validation_word_ids)
        self.validation_tag_ids = np.asarray(self.validation_tag_ids)
        self.testing_word_ids = np.asarray(self.testing_word_ids)
        self.testing_tag_ids = np.asarray(self.testing_tag_ids)

    def construct_training_word_ids(self):
        return self.training_groups_to_ids(self.training_words())

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
        id_by_item = {"__unk__": self.unknown_id, "__pad__": self.padding_id}
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
            self.items(self.groups(self.lines(self.conllu_file_path(self.task, self.language_code, data_type))))))

    def tags_for(self, data_type):
        return self.padded_tags(self.tags(
            self.items(self.groups(self.lines(self.conllu_file_path(self.task, self.language_code, data_type))))))

    def conllu_file_path(self, task, language_code, data_type):
        """
        :param task: "pos", "ner"
        :param language_code: e.g. "da", "en", "ch", etc.
        :param data_type: "training", "validation", "test"
        """
        return f"../../data/{task}/{language_code}/{data_type}.conllu"

    def calculate_sentence_length(self, items):
        longest_sentence = max(items, key=len)
        return len(longest_sentence)

    def padded_words(self, words):
        return [self.pad(group, "__pad__", self.sentence_length) for group in words]

    def padded_tags(self, tags):
        return [self.pad(group, "__pad__", self.sentence_length) for group in tags]

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
