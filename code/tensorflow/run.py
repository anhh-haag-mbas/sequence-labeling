from model import TensorFlowSequenceLabelling

conf = {
    "framework": "tensorflow",
    "crf": False,
    "language": "da",
    "optimizer": "adam",
    "learning_rate": 0.1,
    "dropout": 0,
    "task": "pos",
    "batch_size": 8,
    "data_dir": "../../data/",
    "seed": 5123,
    "epochs": 1,
    "patience": None
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


# langs = ["de", "nl", "ja", "en", "sv", "zh", "sk", "ar", "el"]
# confs = [lang(l) for l in langs]
# confs_crf = [crf(lang(l)) for l in langs]

de = [lang("de"), crf(lang("de")), crf(ner(lang("de")))]
ja = [lang("ja"), crf(lang("ja")), crf(ner(lang("ja")))]

for conf in [ner(lang("de"))]:
    labelling = TensorFlowSequenceLabelling(conf)
    res = labelling.run()
    print(res)
    print()
