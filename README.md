Tarte
=======================

A secondary layer for pie for disambiguation

# What it aims to do

- This tagger is supposed to come as a secondary layer for lemma that should be disambiguated.
- Its core object (`Tarte`) should filter things that need to be disambiguated
- Its training capacities should reorganize a training set so that it dispatch training samples across all sets and
it should not care about sample not containing unambiguous tokens.
- It takes POS, lemma context and form characters into the network to predict the disambiguated form.

# Notes

- Given that not all sentences will have things to disambiguate, pretraining vector might be an important task. It
 is [possible with PyTorch](https://stackoverflow.com/questions/49710537/pytorch-gensim-how-to-load-pre-trained-word-embeddings)
 to load Gensim data easily. This would **require** to generate temps file where lemma AND pos are fed to 
 fake sentences.