#-*- coding: utf-8 -*-
import pickle


with open("../data/corpus_train.pkl", "rb") as f:
    sentences = pickle.load(f)


DIMENSION = 300 # word vector의 차원
WINDOW = 5 # word2vec window size
SG = 1 # 0 : CBOW, 1: Skip-gram
MIN_COUNT = 5 # 학습에 이용할 단어의 corpus 내 최소 등장 횟수
EPOCH = 50 # 학습 반복 수


from gensim.models import Word2Vec
model = Word2Vec(sentences, size=DIMENSION, window=WINDOW, sg=SG, min_count=MIN_COUNT, iter=EPOCH, workers=7)
model.save("../model/wordvector_w2v.model")
