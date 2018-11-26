#-*- coding: utf-8 -*-
from gensim.models import Word2Vec
from glove import Glove

g = Glove.load("../model/wordvector_glove.model")
w2v = Word2Vec.load("../model/wordvector_w2v.model")

word = "쓰레기/Noun"

print(g.most_similar(word, number=10))
print("=" * 20)
print(w2v.most_similar(word))

