import dynet as dy
import numpy as np
import ipdb

from numpy.random import randint

training_size   = 10056
test_size       = 10

def to_bitstring(num):
    return [num >> i & 1 for i in range(31, -1, -1)]

def input_generator(min_value, max_value, training_size):
    return [randint(min_value, max_value) for _ in range(training_size)]

def to_output(input):
    is_odd = 1 if (input % 2 == 1) else 0
    is_negative = 1 if input < 0 else 0
    return [is_odd, is_negative]

def print_bitstring(bitstring):
    print("".join(list(map(str, bitstring))))

int_min = -pow(2, 31)
int_max = pow(2, 31) - 1

input_training_values   = input_generator(int_min, int_max, training_size)
input_test_values       = input_generator(int_min, int_max, test_size)

input_training = list(map(to_bitstring, input_training_values))
label_training = list(map(to_output, input_training_values))

input_test = list(map(to_bitstring, input_test_values))
label_test = list(map(to_output, input_test_values))

ipdb.set_trace()

model = dy.ParameterCollection()
trainer = dy.SimpleSGDTrainer(model)

# Layer 1 
W1 = model.add_parameters((20, 32))
b1 = model.add_parameters(20)

# Layer 2
W2 = model.add_parameters((2, 20))
b2 = model.add_parameters(2)

x = dy.vecInput(32)
y = dy.vecInput(2)

layer1 = dy.logistic((W1 * x) + b1)
y_pred = dy.logistic((W2 * layer1) + b2)
loss = dy.binary_log_loss(y_pred, y)

print("Training...")
for i in range(training_size):
    x.set(input_training[i])
    y.set(label_training[i])
    print(f"loss = {loss.value()}")
    loss.backward()
    trainer.update()

dy.renew_cg()

x = dy.vecInput(32)
layer1 = dy.logistic((W1 * x) + b1)
y_pred = dy.logistic((W2 * layer1) + b2)

for i in range(test_size):
    x.set(input_test[i])
    print_bitstring(input_test[i])
    print(f"Result = {y_pred.value()}")
