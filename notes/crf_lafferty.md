## The text
https://repository.upenn.edu/cgi/viewcontent.cgi?article=1162&context=cis_papers

From 2001

## What is it about?
Presentation of the conditional random fields framework, a discriminative model for sequence level tagging.
The mathematical description and two training algorithms.

How CRF solves the "Label Bias Problem" which other statistical frameworks have.

The label bias problem is how sometimes the model ignores some of the input, because of the way the model is built.
This means that the probability between "rip" and "rop", might be equal, even though "rip" is observed. (As far as I understand)

Comparison between token level, generative models such as Hidden Markov Models(HMM), and Maximum Entropy Markov Models(MEMM).

## What are the findings?
CRF performs better than both the generative models, when the data is "mostly second order". 
It typically outperforms the MEMM any input.

The CRF loss function is convex and it is thus possible to guarantee convergance to a global minimum.

## (What does these things mean?)
Half the paper more or less 