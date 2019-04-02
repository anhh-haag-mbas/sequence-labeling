## The text
http://aclweb.org/anthology/C18-1327

From 2018

## What is it about?
Reproduction of results are a tricky problem, so they implement the best taggers to compare the. 
Referencing the scores of other papers to compare to isn't ideal.
Many references to other papers we have looked at.

Compares CNN vs LSTM, softmax vs CRF, with and without character featres on 3 sequence labeling tasks (POS, Chunking, and NER)

## What are the findings?
Similar results to many existing reports, Reimers and Gurevych has noticably differering results.


## Notable things
"For example, most work observes that stochastic gradient descent (SGD) gives best performance on NER task (Chiu and Nichols, 2016; Lample et al., 2016; Ma and Hovy, 2016), while Reimers and Gurevych (2017b) report that SGD is the worst optimizer on the same datasets."
Preprocessing, features, hyperparameters, evaluation and even hardware environment can have impact on the results.
SGD turns out to be the best optimizer.
Difference between CNN and LSTM in scores is insignificant, but CNN is faster to decode than LSTM.
There is a difference in scores between running on CPU vs GPU.

## (What does these things mean?)