#-*- coding: utf-8 -*-
import pickle

from glove import Corpus, Glove


with open("../data/corpus_train.pkl", "rb") as f:
    corpus = pickle.load(f)


glove_corpus = Corpus()
glove_corpus.fit(corpus, window=5)


DIMENSION = 300 # target word vector dimension
LEARNING_RATE = 0.05 # learning rate for SGD
WEIGHTING_FUNC_PARAM = 0.75 # glove parameter


model = Glove(no_components=DIMENSION, learning_rate=LEARNING_RATE, alpha=WEIGHTING_FUNC_PARAM)
model.fit(glove_corpus.matrix, epochs=10, no_threads=4)
model.add_dictionary(glove_corpus.dictionary)
model.save("../model/model_glove")

model.most_similar("영화/Noun")
