## Parameters
Language
Variance                (Random initializations, sampling)
Models                  (LSTM, bi-lstm, + crf, non-nn)
Hidden dense layers
Number of LSTM layers   (1,2,3 or just argue for one)
Training algorithms / Optimizer     (SGD, etc., Reimer & Greymur)
Embeddings              (Pretrained, glove (is it better since it's only en), fasttext, polyglot, task sepecific, etc.)
Feature representations (word, character, byte(why should it work beter), barbara)
Frameworks              (dynet, tf, pytorch)
Loss function           (evt. barbara)
    Regularization      (with or without)
Dropout                 (0, 10%, 50%, 90%, evt 50% standard)
mini-batches            (Reimers & Greymur)
epochs                  (Train untill converges??)
learning rates          (number + decreasing or static)
Embeddings Input Dimensionality / Dictionary size    ()
Embedding output Dimensionality   ( )
    Eventuelt kun sammenlign store 
    Ikke kun tage laveste fælles nævner
Lemma vs Form

Remember to lock down the testing environment and document it thorougly
Should we use fixed seeds?

## Testing
Speed 
Accuracy
Training speed
Convergence time (number of training sets before overfitting)
Standard deviation


## What range of tests do we work with? Which parameters do we limit ourselves on
