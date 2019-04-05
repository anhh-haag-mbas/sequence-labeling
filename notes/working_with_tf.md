# Notes on working with TF

## Availability of information

It is incredibly easy to find guides/blog posts/papers on how to use the Keras interface for TF. It is furthermore incredibly easy to find code examples and "experimental" designs, such as CRF/LSTM, implementations for TF/Keras.

Finding guides/help for TF, and NOT Keras, is quite difficult.

## High-level nature of Keras

Because of the high level nature of Keras, a lot of implementation details are hidden. This is fine if you are already perfectly familiar with the theory, but makes it harder to understand what the framework is actually doing if you are only somewhat familiar with the theory behind the different things  (LSTM, CRF, MLP, etc.).

Contrast this with other frameworks or just with using TF directly, where everything isn't conviently wrapped in high level "layers". Using these you are forced to understand what is actually going on, at least more than with Keras, just to get any result.

Coupling the high level nature of Keras with the abundance of guides, means that you could actually get started with working code within your given area, without understanding any of the code or theory.

That is undesirable.

On the other hand if you are intimatley familiar with the theory working with TF must be a breeze, as it abstracts all of the technical details away.

Working with non-standard "layers" is also quite easy, but does require a deeper understanding of how TF works.

## Workflow

I have worked a lot by finding examples on various blog posts and then copy pasting them into a jupyter notebook.

I have then experimented with tweaking the examples and writing them together to gain a deeper understanding of TF works. I would often come across features of TF that I did not know existed, which would then prompt me to read the TF docs on that feature.

This allowed me to quite quickly get working results, but it limited just how deeply I got to understand the theory. That is because I relied on existing implementations of bi-lstm/crf, and the way I got started was by looking at how other people had already tackled the problems.

Had I developed the lstm and crf by hand, I would have been forced to gain a much much deeper understanding of how they worked and of how TF worked. This would however have required a much greater time dedication.

It was thus a weighting between time and deep understanding of TF/Keras/CRF/LSTM.
