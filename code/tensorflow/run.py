from model import TensorFlowSequenceLabelling

conf = {
    "framework": "tensorflow",
    "model": "bi-lstm",
    "language": "da",
    "optimizer": "sgd",
    "learning rate": 0.1,
    "embedding type": "task specific",
    # "embedding type": "polyglot",
    "embedding dimensions": 64,
    "dropout": 0,
    "task": "pos",
    "batch_size": 1,
    "repeat": 2
}

labelling = TensorFlowSequenceLabelling(conf)
labelling.quick = True
print(labelling.run())