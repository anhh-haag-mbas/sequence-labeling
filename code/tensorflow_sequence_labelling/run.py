from model import TensorFlowSequenceLabelling

conf = {
    "framework": "tensorflow",
    "crf": False,
    "language": "ru",
    "optimizer": "sgd",
    "learning_rate": 0.1,
    "dropout": 0.5,
    "task": "pos",
    "batch_size": 32,
    "data_root": "../../data/",
    "seed": 5123,
    "epochs": 50,
    "patience": 3
}


def lang(l):
    c = conf.copy()
    c["language"] = l
    return c


def crf(c):
    c = c.copy()
    c["crf"] = True
    return c


def ner(c):
    c = c.copy()
    c["task"] = "ner"
    return c


# langs = ["da", "no", "ru", "hi", "ur", "ja", "ar"]
# confs = [lang(l) for l in langs]
# confs_crf = [crf(lang(l)) for l in langs]

labelling = TensorFlowSequenceLabelling(lang("da"))
res = labelling.run()
print(f"total_values: {res['total_values']}")
print(f"total_errors: {res['total_errors']}")
print(f"total_acc: {res['total_acc']}")
print(res)
print()

# de = [lang("da"), crf(lang("da")), crf(ner(lang("da")))]
# ja = [lang("ja"), crf(lang("ja")), crf(ner(lang("ja")))]
#
# for conf in [ner(lang("da")), lang("da")]:
#     labelling = TensorFlowSequenceLabelling(conf)
#     res = labelling.run()
#     print(res)
#     print()
