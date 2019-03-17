## The text
http://www.aclweb.org/anthology/P16-2067

https://github.com/bplank/bilstm-aux (Code)

From 2016

## What is it about?
Bi-lstm for POS tagging on 22 languages, at 3 different levels of granularity (words, characters, and bytes) and combinations of these

How much training does a bi-LSTM require compared to other standard POS taggers

An auxiliary loss on POS and the log frequency of the next word

Comparisons with HMM and a CRF taggers

## What are the findings?
Using word embeddings is great. 

The word + character models did very well, including the auxiliary loss did best overall (12/22 languages), succeding in it's task in increasing the accuracy on rare words

The bi-LSTM model didn't need much more data than the HMM tagger and dominated the CRF tagger

The bi-LSTM and the HMM were affected similarly by noise in the input, only when more than 30% of the labels were corrupted did the bi-LSTM fall off compared to the HMM model

## (What does these things mean?)
Gold segmentation

We  also  re-port accuracies on WSJ (45 POS) using the standard splits (Collins, 2002;  Manning, 2011)
