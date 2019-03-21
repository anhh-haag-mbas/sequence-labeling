import dynet as dy
import numpy as np
import ipdb

# reset the global cg
dy.renew_cg()
# create parameter collection
m = dy.ParameterCollection()

# add parameters to parameter collection
pW = m.add_parameters((10,30))
pB = m.add_parameters(10)
lookup = m.add_lookup_parameters((500, 10))
print("Parameters added.")

# create trainer
trainer = dy.SimpleSGDTrainer(m)

# Regularization is set via the --dynet-l2 commandline flag.
# Learning rate parameters can be passed to the trainer:
# alpha = 0.1  # learning rate
# trainer = dy.SimpleSGDTrainer(m, e0=alpha)

# function for graph creation
def create_network_return_loss(inputs, expected_output):
    """
    inputs is a list of numbers
    """
    ipdb.set_trace()
    dy.renew_cg()
    emb_vectors = [lookup[i] for i in inputs]
    net_input = dy.concatenate(emb_vectors)
    net_output = dy.softmax( (pW*net_input) + pB)
    loss = -dy.log(dy.pick(net_output, expected_output))
    return loss

# function for prediction
def create_network_return_best(inputs):
    """
    inputs is a list of numbers
    """
    ipdb.set_trace()
    dy.renew_cg()
    emb_vectors = [lookup[i] for i in inputs]
    net_input = dy.concatenate(emb_vectors)
    net_output = dy.softmax( (pW*net_input) + pB)
    return np.argmax(net_output.npvalue())



# train network
for epoch in range(5):
    for inp,lbl in ( ([1,2,3],1), ([3,2,4],2) ):
        loss = create_network_return_loss(inp, lbl)
        print(loss.value()) # need to run loss.value() for the forward prop
        loss.backward()
        trainer.update()

print(f'Predicted smallest element among {[1, 2, 3]} is {create_network_return_best([1,2,3])}')
