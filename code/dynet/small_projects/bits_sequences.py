import dynet as dy
import ipdb
from random import randint

# Utility 
def bitstring_to_text(bitstring):
    return ("".join(list(map(str, bitstring))))

def bitstring_to_num(bitstring):
    return int(bitstring_to_text(bitstring), 2)

def output_to_text(output):
    idx = output.index(max(output))
    if idx == 0: return "lower"
    if idx == 1: return "equal"
    if idx == 2: return "greater"
    return "output size greater than expected size of 3"

#Input / label generation
def input_generator(min_value, max_value, training_size):
    return [randint(min_value, max_value) for _ in range(training_size)]

def to_bitstring(num):
    return [num >> i & 1 for i in range(31, -1, -1)]

def to_output(next_prev_pair):
    next, prev  = next_prev_pair
    is_lower    = 1 if next < prev else 0
    is_even     = 1 if next == prev else 0
    is_greater  = 1 if next > prev else 0

    return [is_lower, is_even, is_greater]

int_min = -pow(2, 31)
int_max = pow(2, 31) - 1

training_size = 2870
sequence_length = 10
test_size = 10

training_input_values = input_generator(int_min, int_max, training_size)
test_input_values     = input_generator(int_min, int_max, test_size)

training_input = list(map(to_bitstring, training_input_values))
training_output = [[0,0,0]] + list(map(to_output, zip(training_input_values[1:], training_input_values)))

test_input = list(map(to_bitstring, test_input_values))
test_output = [[0,0,0]] + list(map(to_output, zip(test_input_values[1:], test_input_values)))

# Model building
model = dy.ParameterCollection()
trainer = dy.SimpleSGDTrainer(model)

OUTPUT_DIM = len(to_output((0, 0)))
INPUT_DIM = len(to_bitstring(0)) + OUTPUT_DIM

HIDDEN_DIM = 16

lstm = dy.LSTMBuilder(
        layers = 1, 
        input_dim = INPUT_DIM, 
        hidden_dim = HIDDEN_DIM, 
        model = model)

weight = model.add_parameters((OUTPUT_DIM, HIDDEN_DIM))
bias = model.add_parameters(OUTPUT_DIM)

def do_one_sequence(sequence, outputs):
    # Creating the computational graph
    dy.renew_cg()
    s = lstm.initial_state()

    input = dy.vecInput(INPUT_DIM)
    loss = []
    prev_output = [0, 0, 0]
    #ipdb.set_trace()
    for bitstring, output in zip(sequence, outputs):
        input.set(bitstring + prev_output)
        s = s.add_input(input)
        prob_dist = dy.softmax(weight * s.output() + bias)
        prev_output = prob_dist.value()
        loss.append(-dy.log(dy.pick(prob_dist, output.index(max(output)))))

    return dy.esum(loss)

for i in range(int(training_size / sequence_length)):
    start = i * sequence_length
    sequence = training_input[start:start+sequence_length]
    outputs = training_output[start:start+sequence_length]
    outputs[0] = [0, 0, 0]
    loss = do_one_sequence(sequence, outputs)
    loss_value = loss.value()
    loss.backward()
    trainer.update()

    if i % 10 == 0: 
        print("%.10f" % loss_value)




sequence = test_input
s = lstm.initial_state()

input = dy.vecInput(INPUT_DIM)
loss = []
prev_output = [0, 0, 0]
for i, bitstring in enumerate(sequence):
    input.set(bitstring + prev_output)
    s = s.add_input(input)
    prob_dist = dy.softmax(weight * s.output() + bias)
    prev_output = prob_dist.value()
    if i > 0:
        current = bitstring_to_num(sequence[i]) 
        prev =  bitstring_to_num(sequence[i-1]) 
        output_text = output_to_text(prev_output)
        result = "error"
        if "lower" == output_text:
           result = current < prev 
        if "equal" == output_text:
           result = current == prev 
        if "greater" == output_text:
           result = current > prev 

        print(f"{current}\t {output_text} than\t{prev} ({result})")

print ("Remember ! before s and stuff")

ipdb.set_trace()
