## The text
https://arxiv.org/pdf/1508.01991.pdf

From 2015

## What is it about?
THE Paper(I think) which introduced the idea of using a CRF on top of a BI-LSTM model.

A systematic comparison between 5 different models:
 * LSTM
 * BI-LSTM
 * CRF 
 * LSTM-CRF
 * BI-LSTM-CRF

Contains small explanations of LSTM, CRF, NER and POS tagging as well.

## Notes
They used a batch size of 100, but for them the batch size was the combined length of the sentences. 
ie. a batch could consist of 10 sentence of avg. length 10, or 2 of average length 50.
(I am assuming the length means words number of words, but not sure, would make better sense than characters)

The training for three tasks require less than 10 epochs to converge and it in general takes less than a few hours.

## What are the findings?
Their model outperforms most of the other state of the art models.

BI-LSTM-CRF is robust and has less dependency on word embeddings.

Non-linear architectures offers no benefits in a high dimensional discrete feature space.

## (What does these things mean?)
"Features connection tricks"

"We have 401K, 76K, and 341K features extracted for POS,chunking and NER data sets respectively" (How so many?)