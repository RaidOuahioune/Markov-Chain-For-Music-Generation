# This a Markov chain model built from scratch in the goal of music generation(piano pitches)

The models uses the current pitch and its duration(ie the current state only ) to predict the next picth and its duration also, meaning the model is equivalent to a markovify model with state_order=1

So in our model the state consists of a tuple of 2 things:

1. Pitch Type: For example C5 , D4 ...

2. The duration: 1,2 ...


The model takes an ordered sequence of pitches and calculate the frequency of each 2 consecutive pairs to infer the transition matrix from one state to another.

# He is a recording for the original music
