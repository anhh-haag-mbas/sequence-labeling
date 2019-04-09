from model import TensorFlowSequenceLabelling

conf = {
    "framework": "tensorflow",
    "crf": "bi-lstm",
    "language": "da",
    "optimizer": "sgd",
    "learning_rate": 0.1,
    "dropout": 0.5,
    "task": "pos",
    "batch_size": 100,
    "data_dir": "../../data/",
    "seed": 5123,
    "epochs": 1,
    "patience": None
}

labelling = TensorFlowSequenceLabelling(conf)
labelling.quick = True
print(labelling.run())