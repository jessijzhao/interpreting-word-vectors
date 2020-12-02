# Interpretable Word Vector Subspaces

Presented at the ICLR 2020 workshop on Machine Learning in Real Life (ML-IRL) on April 26 2020. Link to paper [here](https://drive.google.com/file/d/1ZlJxQxn6Fj6utqI4zgaLqbyLDQ9DSFds/view).

## Abstract

Natural Language Processing relies on high-dimensional word vector representations, which may reflect biases in the training text corpus. Identifying these biases by finding corresponding interpretable subspaces is crucial for achieving fair decisions. Existing works have adopted Principal Component Analysis (PCA) to identify subspaces such as gender but fail to generalize efficiently or provide a principled methodology. We propose a framework for existing PCA methods and considerations for optimizing them. We also present a novel algorithm for finding topic subspaces more efficiently and compare it to an existing approach.

## Contents

- identifying_subspaces.ipynb (TBA)

- [bolukbasi_et_al.ipynb](bolukbasi_et_al.ipynb) reproduces a subset of the results from
[Man is to Computer Programmer as Woman is to
Homemaker? Debiasing Word Embeddings](http://papers.nips.cc/paper/6228-man-is-to-computer-programmer-as-woman-is-to-homemaker-debiasing-word-embeddings.pdf) by
Tolga Bolukbasi, Kai-Wei Chang, James Zou, Venkatesh Saligrama, and Adam Kalai

- [helpers.py](helpers.py) implements the functions used to reproduce Bolukbasi et al.'s results

- [data/figure_7_words.txt](data/figure_7_words.txt) stores words for reproducing Fig. 7 from Bolukbasi et al.'s paper

- [data/gender_specific_words.txt](data/gender_specific_words.txt) stores seed words for training a classifier that distinguishes gender-specific words from gender-neutral words
