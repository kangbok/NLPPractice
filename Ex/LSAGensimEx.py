#-*- coding: utf-8 -*-
import pickle

from gensim.corpora import Dictionary
from gensim.models import LsiModel

with open("../data/corpus_test.pkl", "rb") as f:
    corpus = pickle.load(f)

corpus_dictionary = Dictionary(corpus)
corpus = [corpus_dictionary.doc2bow(text) for text in corpus]

CORPUS = corpus
ID2WORD = corpus_dictionary
NUM_TOPICS = 200

lsi = LsiModel(corpus=corpus, num_topics=NUM_TOPICS, id2word=ID2WORD)

topic_word_matrix = lsi.get_topics()
sigma_matrix = lsi.projection.s # SVD에서 sigular value들로 이루어진 sigma matrix. 편의를 위해 형태는 그냥 k vector
# document-topic matrix를 구해주지 않는다.

new_doc = ["영화/Noun", "재미/Noun"] # 새로운 document
new_doc_bow = corpus_dictionary.doc2bow(new_doc) # 새로운 document의 bag of words
vec_lsi = lsi[new_doc_bow] # 새로운 document를 LSI 공간으로 사상.
print(vec_lsi)


