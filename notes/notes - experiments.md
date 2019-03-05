## Parameters
Language                (Find argument/paper with good argument)
Variance                (Random initializations, sampling)
Models                  (LSTM, bi-lstm, + crf, non-nn)
Hidden dense layers
Number of LSTM layers   (1,2,3 or just argue for one)
Training algorithms / Optimizer     (SGD, etc., Reimer & Greymur)
Embeddings              (Pretrained, glove (is it better since it's only en), fasttext, polyglot, word2vec task sepecific, etc.)
Feature representations (word, character, byte(why should it work beter), barbara)
Frameworks              (dynet, tf, pytorch)
Loss function           (evt. barbara)
    Regularization      (with or without)
Dropout                 (0, 10%, 50%, 90%, evt 50% standard)
mini-batches            (Reimers & Greymur)
epochs                  (Train untill converges?? / Use model at the time of convergence and continue)
learning rates          (number + decreasing or static)
Embeddings Input Dimensionality / Dictionary size    (unlimited og fair)
Embedding output Dimensionality   (unlimited og fair)
    Eventuelt kun sammenlign store 
    Ikke kun tage laveste fælles nævner
Lemma vs Form

Remember to lock down the testing environment and document it thorougly
Should we use fixed seeds?

## Testing
Speed 
Accuracy
Standard deviation (Calculated after)
Training speed (Total time)
Convergence time (number of training sets before overfitting)



## What range of tests do we work with? Which parameters do we limit ourselves on

Language        10
Variance        Sample
Models          3 (2)
Layers          3
Optimizers      2
Embeddings      3
Featues         3 ()
Loss functions  3 (1)
Frameworks      3 (1)
Dropout         3 (1)
Mini-batches    3 (1)
Epochs          2 (1)
Learning rates  4 (2)
Dictionary size 2 (1)
Output          2 (1)
Lemma vs form   2 (1)

result      =   2 * 3 * 3 * 3 * 2 * 3 * 3 * 10

