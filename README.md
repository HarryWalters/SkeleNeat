# SkeleNeat
My entry in the NeurIPS 2018 "AI For Prosthetics Challenge". I implemented the neuroevolving algorithm NEAT, only using Python's base packages.

This work is directly based on the 2002 paper "Evolving Neural Networks through Augmenting Topologies", by Kenneth Stanley and Risto Miikkulainen. NEAT essentially constructs 'species' of simple neural networks which build upon themselves over time, emulating biological mutation and crossbreeding.

I recognized that most participants were using RL algorithms like TRPO and A3C. After watching youtuber Sethbling use an older reinforcement learning technique NEAT on a high-dimensional task, I decided to implement my own version of the algorithm.

An interesting thing to note is the non-neccesity of third-party packages. To test the extent of my data structure skills, I built the enitre implementation using base packages (primarily lists; to account for augmenting architectures).


# References
Stanley, K. O. & Miikkulainen, R. (2002). Evolving Neural Networks through Augmenting Topologies. The MIT Press Journals.
http://nn.cs.utexas.edu/downloads/papers/stanley.ec02.pdf


Sethbling. (2015). MarI/O - Machine Learning for Video Games. YouTube.
https://www.youtube.com/watch?v=qv6UVOQ0F44
