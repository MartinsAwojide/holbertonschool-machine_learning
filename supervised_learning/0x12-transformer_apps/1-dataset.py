#!/usr/bin/env python3
"""Class dataset"""
import tensorflow.compat.v2 as tf
import tensorflow_datasets as tfds

tf.compat.v1.enable_eager_execution()


class Dataset:
    """loads and preps a dataset for machine translation"""
    def __init__(self):
        """Class initializer"""
        examples, metadata = tfds.load('ted_hrlr_translate/pt_to_en',
                                       with_info=True,
                                       as_supervised=True)
        self.data_train, self.data_valid = examples['train'], \
            examples['validation']

        self.tokenizer_pt, self.tokenizer_en = self.tokenize_dataset(
            self.data_train)

    def tokenize_dataset(self, data):
        """
        creates sub-word tokenizers for our dataset
        Args:
            data:  is a tf.data.Dataset whose examples are formatted
             as a tuple (pt, en)
            pt is the tf.Tensor containing the Portuguese sentence
            en is the tf.Tensor containing the corresponding English sentence

        Returns: tokenizer_pt, tokenizer_en
        tokenizer_pt is the Portuguese tokenizer
        tokenizer_en is the English tokenizer

        """
        tokenizer_pt = tfds.features.text.SubwordTextEncoder.build_from_corpus(
            (pt.numpy() for pt, en in data), target_vocab_size=2 ** 15)
        tokenizer_en = tfds.features.text.SubwordTextEncoder.build_from_corpus(
            (en.numpy() for pt, en in data), target_vocab_size=2 ** 15)
        return tokenizer_pt, tokenizer_en

    def encode(self, pt, en):
        """
        encodes a translation into tokens
        Args:
            pt: is the tf.Tensor containing the Portuguese sentence
            en: is the tf.Tensor containing the corresponding English sentence

        Returns: pt_tokens, en_tokens
        pt_tokens is a tf.Tensor containing the Portuguese tokens
        en_tokens is a tf.Tensor containing the English tokens
        """
        pt_tokens = [self.tokenizer_pt.vocab_size] + self.tokenizer_pt.encode(
            pt.numpy()) + [self.tokenizer_pt.vocab_size + 1]

        en_tokens = [self.tokenizer_en.vocab_size] + self.tokenizer_en.encode(
            en.numpy()) + [self.tokenizer_en.vocab_size + 1]

        return pt_tokens, en_tokens
