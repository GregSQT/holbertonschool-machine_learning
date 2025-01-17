#!/usr/bin/env python3
"""fasttext model"""
import gensim


def fasttext_model(
        sentences,
        size=100,
        min_count=5,
        negative=5,
        window=5,
        cbow=True,
        iterations=5,
        seed=1,  # Set seed to 1
        workers=1):
    """
    Train a FastText model on the given sentences.

    Args:
        sentences (iterable): The sentences to train the model on.
        size (int): The dimensionality of the word vectors.
        min_count (int): The minimum number of times a word must appear
        to be included in the vocabulary.
        negative (int): The number of negative samples to use during training.
        window (int): The maximum distance between the current and predicted
        word within a sentence.
        cbow (bool): Whether to use the Continuous Bag of Words (CBOW)
        architecture If False, use Skip-gram.
        iterations (int): The number of iterations (epochs) over the corpus.
        seed (int): The seed for the random number generator.
        workers (int): The number of worker threads to train the model.
    Returns:
        FastText: The trained FastText model.
    """
    sg = 0 if cbow else 1
    model = gensim.models.FastText(
        sentences,
        vector_size=size,
        window=window,
        min_count=min_count,
        sg=sg,
        seed=seed,
        negative=negative,
        workers=workers)
    model.build_vocab(sentences)
    model.train(
        sentences,
        total_examples=model.corpus_count,
        epochs=model.epochs)
    return model
