## The text
http://aclweb.org/anthology/D17-1035

https://github.com/UKPLab/emnlp2017-bilstm-cnn-crf (Code)

From 2017

## What is it about?
A demonstration that a single performance score is insufficient when comparing non-deterministic approaches

Evaluates 50.000 different LSTM networks over 5 different sequence tagging tasks

Multiple iterations of initialization and training to account for the random initialization

Evaluates on the following parameters using random sampling:
 * Pre-trained word embeddings
 * Character Representation
 * Optimizer
 * Gradient clippign and normalization 
 * Tagging schemes
 * Different dropout methods / rates
 * ClassifierPre-trained word embeddings
 * LSTM layers
 * Recurrent units
 * Mini-batch sizes

975 configurations were sampled for both softmax classifier and CRF classifier for a total of 1950 trained networks

## What are the findings?
Using the development set to decide on the best model for the test set is bad

Using the score distribution to compare techniques is superior to single comparisons.
This method reduces the risk of rejecting good ideas or accepting bad or worse ideas

## Findings demonstrations

### Classifier
BiLSTM-CRF is better than softmax for 4 out of the 5 tasks.
It has a lower standard deviation which makes it less dependent on hyperparameters and the random number generator

The softmax classifier benefits a lot from multiple LSTM layers whereas the CRF is less susceptible

### Optimizer
SGD is very sensitive to learning rates

The optimizers RMSProp, Adam and Nadam scored higher the their standard deviations were smaller than the other optimizers

The best performance was acchieved by Nadam

### Word Embeddings
Komninos and Manandhar (2016) best for POS, Entities and Events

Levy and Goldberg (2014) best for Chunking and dependency 

Note: 
The underlying algorithms for creating these embeddings were not evaluated
and since the corpora are not comparable these findings doesn't imply one method is better than another

### Character Representation
Only relevant for POS, chunking and events, but nothing conclusive can be said for CNN vs LSTM

### Gradient Clipping and Normalization
No improvements for the different thresholds on any tasks when doing gradient clipping

Gradient normalization gave large improvements, the best threshold was 1

### Dropout
Variational dropout is best by a large margin

Dropout on both the output of the LSTM and on the recurrent units was superior to just one, the other, or none

### Tagging schemes
On par for 4 / 5 tasks, BIO outperformed IOBES significantly on the entities task

### LSTM layers and recurrent units
2 stacked LSTM-layers generally did best

The number of recurrent units didn't have a significant impact

### Mini-batch size
For small training sets, mini-batches between 1-16 all work well

For larger training sets, sizes between 8-32 works well

Mini-batches of 64 are pretty bad

## (What does these things mean?)
Brown-Forsythe test

## Related work
Ma and Hovy (2016) & Lample et al. (2016) - the NER systems under investigation

Erhan et al. (2010) showed for the MNIST handwritten digit recognition task that different random seeds result in largely varying performances. 
They noted further that with increasing depth of the neural network, the probability of finding poor local minima increases.

As (informally) defined by Hochreiter and Schmid-huber (1997a), a minimum can be flat, 
where the error function remains approximately constant for a large connected region in weight-space, 
or it can be sharp, where the error function increases rapidly in a small neighborhood of the minimum. 

A sharp minimum usually depicts poorergeneralization capabilities, 
as a slight variation re-sults in a rapid increase of the error function. 
On the other hand, flat minima generalize better on new data (Keskar et al., 2016).

Pascanu et al. (2013) gives justifiation for Gradient normalization