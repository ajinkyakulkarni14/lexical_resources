#!/usr/bin/env python3

# -*- coding: utf-8 -*-

import collections
import math
import random

import numpy as np


def unpair_lexicon(lexicon_pairs) :
    """
        This function turns a list of pairs into a pair of lists.
        Supposing for instance the list:
            [('chat', 'cat'), ('chien', 'dog'), ('musique', 'music')]
        it will return:
            ['chat', 'chien', 'musique'], ['cat', 'dog', 'music']
    """
    return zip(*lexicon_pairs)


def split_train_text(examples, shuffle=True):
    """
        This function splits the list `examples` into two sublists. The first
        will contain 90% of the elements, the second 10%. If shuffle is set to
        True, it will first shuffle the list randomly.
    """
    if shuffle:
        random.shuffle(examples)
    prop = math.floor(len(examples) * 9 / 10)
    return examples[:prop], examples[prop:]


def compute_transformations(WE_fr_paired, WE_en_paired):
    """
        This function computes the approximate transformations for two word
        embedding spaces, based on their sub-matrices passed as parameters.
        It expects that the word corresponding to the vector in row i in any of
        the two matrices is the translation of the word corresponding to the
        vector in row i of the other.
        It returns two matrices, corresponding to linear transformations, to be
        applied to the entire embedding spaces: if WE_fr_paired is a subset of
        WE_fr and WE_en_paired is a subset of WE_en, then the first
        transformation `u` should be applied to WE_fr, and the second `vT.T`
        should be applied to WE_en.
    """
    P = WE_fr_paired.T @ WE_en_paired
    u,s,vT = np.linalg.svd(P)
    return u, vT.T


def list_to_matrix(list_of_vectors):
    """
        This function turns a list of vectors into a matrix where vectors are
        stored as row, with indices matching the original list
    """
    return np.matrix(list_of_vectors)


def read_vector_from_file(file_path, index):
    """
        This function retrieves a single vector from a file in word2vec format,
        based on its index. It might be helpful if your computer cannot keep the
        entire matrix in memory.
    """
    with open(file_path, "r") as istr:
        istr.readline() #skip header
        for i,l in enumerate(istr):
            if i==index:
                return np.array(map(float, l.strip().split()[1:]))


def compute_lookup_embeddings(file_path, with_embedding=True):
    """
        This function computes a lookup and an embedding matrix based on the
        file in word2vec format passed as paramater.
        If with_embedding is False, it will only compute and return the
        lookup.
    """
    with open(file_path, 'r') as istr:
        V, d = map(int, istr.readline().strip().split(' '))
        lu = {}
        if with_embedding:
            em = np.zeros((V,d))
        for i,l in enumerate(istr):
            w, v = l.strip().split(' ', 1)
            lu[w]=i
            if with_embedding:
                em[i,:]=list(map(float, v.split()))
    if with_embedding :
        return lu, em
    return lu


def cosine(vec1, vec2):
    """
        This function computes the cosine between two vectors.
    """
    return vec1.dot(vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))


def k_top(word, lookup, embeddings, k=5) :
    """
        This function retrieves the k most similar words to `word`in the
        embbeding matrix `embeddings`.
    """
    vec = embeddings[lookup[word],:]
    return sorted(
        lookup,
        key=lambda w:cosine(vec, embeddings[lookup[w],:]),
        reverse=True,
    )[:k]


class EmbeddingSpace:
    """
        This class is a container holding a reference to a lookup and a matrix.
        It might help you to produce cleaner code.
        - Create an object using :
            space = EmbeddingSpace(lookup, matrix)
        - Access its lookup using :
            space.lookup
        - Access the embedding matrix using :
            space.matrix
        - Retrieve the vector for the word `word` using :
            space[word]
        You can also build an EmbeddingSpace objecy from a file path using
            space = EmbeddingSpace.from_file(file_path)
        You can also add some specific class methods to compute cosine, retrieve
        the k most similar items, etc.
    """

    def __init__(self, lookup, embedding_matrix) :
        self.lookup = lookup
        self.matrix = embedding_matrix

    def __getitem__(self, word):
        return self.matrix[self.lookup[word],:]

    @classmethod
    def from_file(cls, file_path): # copied from above
        with open(file_path, 'r') as istr:
            V, d = map(int, istr.readline().strip().split(' '))
            lu, em = {}, np.zeros((V,d))
            for i,l in enumerate(istr):
                w, v = l.strip().split(' ', 1)
                lu[w]=i
                em[i,:]=list(map(float, v.split()))
        return cls(lu, em)
