#!/usr/bin/env python3
"""creates and trains a gensim word2vec model"""
from gensim.models import Word2Vec


def word2vec_model(sentences, size=100, min_count=5, window=5, negative=5,
                   cbow=True, iterations=5, seed=0, workers=1):
    """
    creates and trains a gensim word2vec model
    Args:
        sentences: is a list of sentences to be trained on
        size: is the dimensionality of the embedding layer
        min_count: is the minimum number of occurrences of a word for use
        in training
        window: is the maximum distance between the current and predicted
        word within a sentence
        negative: is the size of negative sampling
        cbow: is a boolean to determine the training type; True is for CBOW;
        False is for Skip-gram
        iterations: is the number of iterations to train over
        seed: is the seed for the random number generator
        workers: is the number of worker threads to train the model

    Returns: the trained model
    """
    # source: https://radimrehurek.com/gensim/models/word2vec.html
    model = Word2Vec(size=size, window=window, min_count=min_count,
                     workers=workers, negative=negative, sg=cbow, seed=seed)
    model.build_vocab(sentences)
    # If sentences is the same corpus that was provided to build_vocab()
    # earlier, you can simply use total_examples=self.corpus_count.
    model.train(sentences, total_examples=model.corpus_count,
                epochs=iterations)
    return model
