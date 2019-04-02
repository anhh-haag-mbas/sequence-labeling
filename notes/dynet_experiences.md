My general approach to learning dynet was to ease myself into it, following simple tutorials like an XOR example, and building from there.
With an understanding of how a simple network works I could try to create my own on a toy problem and see if I really understood how stuff worked. 
One of the nice things about Dynet is how close it feels to creating computational graphs. 
Creating parameters and linking them together by applying mathematical operations on them, made it easy to reason about the behavior of the model.
After the small examples it was simple to learn how to work an LSTM layer into the model.
LSTM in dynet hides a lot of the complexity associated with the computations, but the interface was still intuitive to use.
Get the initial state, compute the next state based on input, get the output for the next layer and save the state for the next part of the sequence.
Working out how to add a CRF layer on top was mostly a matter of understanding CRF rather than Dynet. 
Figuring out that what we needed for CRF was simply a transformation matrix, made it obvious that this matrix should just be added asa a parameter similar to other parts of a model.
From there I could take inspiration from a complete model implemented in dynet and reuse their code for the CRF.