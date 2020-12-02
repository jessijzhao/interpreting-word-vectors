'''
Author: jessijzhao
Date: April 14, 2020 (revised December 2020)

Implements helper functions for finding interpretable word vector subspaces
with PCA based on Bolukbasi et al.'s 2016 paper.
'''

import itertools
import numpy as np
import matplotlib.pyplot as plt
import random
import string

from numpy.linalg import norm
from scipy.spatial.distance import cosine as cosine_dist
from sklearn.decomposition import PCA


def cosine(u, v):
    '''
    Returns the cosine similarity between vectors u and v.
    '''
    return 1 - cosine_dist(u, v)



def filter_model(model):
    '''
    model: pre-trained word vector model

    Returns only lower-case words and phrases consisting of fewer than 20
    characters. Discards words with upper-case letters, digits, or
    punctuation.
    '''
    def clean(word):
        '''
        Returns whether the word satisfies the conditions
        '''
        characters = set(string.ascii_lowercase + '_')
        return (all(c in characters for c in word) and len(word) < 20)

    return [w for w in model.vocab.keys() if clean(w)]


def generate_analogies(a, b, model, candidates=[], num=10000, delta=1):
    '''
    a, b: words that constitute the left side of the analogy
    model: pre-trained word vector model
    candidates: pool of words to consider for the right side of the analogy
    num: only consider the top num candidates
    delta: sets threshold for maximum semantic similarity between x and y

    Returns the highest scoring analogies of the form a:b::x:y with their score.
    '''
    if len(candidates) == 0:
        candidates = list(model.vocab.keys())
    candidates = candidates[:num]
    candidates.remove(a)
    candidates.remove(b)

    vec_a, vec_b = model[a], model[b]
    result = []

    for x, y in itertools.combinations(candidates, 2):
        vec_x, vec_y = model[x], model[y]

        # exclude analogies that would score 0
        if norm(vec_x-vec_y) <= delta:
            score = cosine(vec_a-vec_b, vec_x-vec_y)
            if score > 0:
                result.append(('{} : {} :: {} : {}'.format(a, b, x, y), score))
            else:
                result.append(('{} : {} :: {} : {}'.format(a, b, y, x), -1 * score))

    return sorted(result, key=lambda x: x[1], reverse=True)


def center_vectors(defining_sets, model):
    '''
    defining_sets: list of defining sets
    model: pre-trained word vector model

    Returns the centered vectors for each defining set as well as C.
    '''
    def get_C(v):
        '''
        Returns the partial sum for C associated with the defining set.
        '''
        return np.matmul(v.reshape(-1, 1), v.reshape(1, -1)) / len(D)

    # applying PCA to C returns the topic subspace
    C = np.zeros((300, 300))

    # this stores the centered word vectors
    c_vectors = []

    for D in defining_sets:
        c_vec = model[D] - np.sum(model[D], axis=0)/len(D)
        c_vectors += c_vec.tolist()
        C += np.sum(np.apply_along_axis(get_C, 1, c_vec), axis=0)

    return np.array(c_vectors), C


def identify_subspace(defining_sets, model, k=1, compare=False):
    '''
    defining_sets: list of defining sets
    model: pre-trained word vector model
    k: number of dimensions for the topic subspace
    compare: whether to compare the defining sets with the topic subspace

    Returns the topic subspace B of dimension k identified by the defining sets.
    '''
    c_vectors, C = center_vectors(defining_sets, model)
    pca = PCA(n_components=k)
    pca.fit(C)
    B = pca.components_

    if compare:
        for i in range(0, 2*len(defining_sets), 2):
            cosine_similarity = cosine(c_vectors[i]-c_vectors[i+1], B)
            print('{}: similarity = {}'.format(
                defining_sets[i//2],
                round(abs(cosine_similarity), 3)
                )
            )

    return B


def projection(u, v, vector=False):
    '''
    u, v: vector representations
    vector: whether to use scalar or vector projection

    Returns the scalar or vector projection of u onto v.
    '''
    if vector:
        return np.dot(u, v) / norm(v) * v
    else:
        return np.dot(u, v) / norm(v)


def neutralize(words, B, model):
    '''
    words: list of words to neutralize
    B: the topic subspace
    model: pre-trained word vector model

    Returns a dictionary mapping words to neutralized vectors for N.
    '''
    def neutr(w):
        w_b = projection(w, B, vector=True)
        return (w-w_b) / (norm(w-w_b))

    vectors = np.apply_along_axis(neutr, 1, model[words])
    return {words[i]: np.array(vectors[i]) for i in range(len(words))}


def equalize(equality_sets, B, model):
    '''
    equality_sets: sets in which all words should be identical except in B
    B: the topic subspace
    model: pre-trained word vector model

    Returns a dictionary mapping words to equalized word vectors for all words
    in the equality sets.
    '''
    def equal(w):
        w_b = projection(w, B, vector=True)
        w = v + np.sqrt(1 - np.square(norm(v)))
        return w * (w_b - mu_b) / norm(w_b - mu_b)

    words = np.array(equality_sets).flatten()
    vectors = []

    for E in equality_sets:
        mu = np.sum(model[E], axis=0) / len(E)
        mu_b = projection(mu, B, vector=True)
        v = mu - mu_b
        w_new = np.apply_along_axis(equal, 1, model[E])
        vectors += w_new.tolist()

    return {words[i]: np.array(vectors[i]) for i in range(len(vectors))}


def debias(sets, B, model, equal=False, compare=False):
    '''
    sets: list of word sets to debias
    B: the topic subspace
    model: pre-trained word vector model
    equal: whether to equalize or neutralize
    compare: whether to compare the cosine similarity of each set with the
             subspace before and after debiasing

    Returns a dictionary mapping words to debiased vectors.
    '''
    words = np.array(sets).flatten()

    if equal:
        new_model = equalize(sets, B, model)
    else:
        new_model = neutralize(words, B, model)

    if compare:
        print('Cosine Similarity with subspace before and after debiasing:')
        for (x, y) in sets:
            pre = round(abs(cosine(B, model[x]-model[y])), 2)
            post = round(abs(cosine(B, new_model[x]-new_model[y])), 2)
            print('{}: before={}, after={}'.format((x, y), pre, post))

    return new_model


def figure_6(model, k=10, sets=[]):
    '''
    model: pre-trained word vector model
    k: number of components to display
    sets: sets of words to compute variance for

    Applies PCA and plots the explained variance ratio for the top k
    eigenvectors.
    '''
    def get_explained_variance(sets, model):
        '''
        sets: sets of words to perform PCA on
        model: pre-trained word vector model

        Returns the explained variance ratios for each eigenvector.
        '''
        c_vectors, _C = center_vectors(sets, model)
        pca = PCA(n_components=k)
        pca.fit(c_vectors)
        return pca.explained_variance_ratio_

    # randomly sample sets and take the mean explained variance
    if len(sets) == 0:
        variance = []
        for i in range(100):
            sample = random.sample(model.vocab.keys(), k=2*k)
            sets = [(sample[i], sample[-i-1]) for i in range(10)]
            variance.append(list(get_explained_variance(sets, model)))
        variance = np.mean(np.array(variance), axis=0)

    # compute explained variance for given sets
    else:
        variance = get_explained_variance(sets, model)

    plt.bar([i for i in range(k)], variance)
    plt.show()


def figure_7(words, clf, model):
    '''
    words: list of words to plot
    clf: classifier that identifies gender-specific words
    model: pre-trained word vector model

    Plots words against the vector difference between 'he' and 'she' (x-axis)
    and their distance from the decision boundary (y-axis).
    '''
    def he_she_projection(word):
        return projection(model[word], model['he']-model['she'])

    he_she_proj = np.vectorize(he_she_projection)
    plot_x = he_she_proj(words)

    # the signed distance of each word to the decision boundary
    vectors = np.zeros((len(words), 300))
    for i in range(len(words)):
        vectors[i] = model[words[i]]
    plot_y = -1. * clf.decision_function(vectors)

    plt.scatter(plot_x, plot_y, alpha=0)
    plt.axhline(0, 0, 1)
    plt.axvline(0, 0, 1)
    for i, word in enumerate(words):
        plt.annotate(word, (plot_x[i], plot_y[i]))
    plt.axis('off')
    plt.show()
