#-*- coding: utf-8 -*-
import pickle

from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer

with open("../data/corpus_test.pkl", "rb") as f:
    corpus = pickle.load(f)

corpus = [" ".join(sentence) for sentence in corpus]

cv = CountVectorizer()
v_list = cv.fit_transform(corpus)

lda = LatentDirichletAllocation(n_components=5, random_state=0)
r = lda.fit_transform(v_list)

print("a")
