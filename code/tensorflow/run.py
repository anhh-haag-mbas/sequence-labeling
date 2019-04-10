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

labelling = TensorFlowSequenceLabelling(conf)
res = labelling.run()
print(res)