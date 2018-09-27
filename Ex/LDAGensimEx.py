#-*- coding: utf-8 -*-
import pickle

from gensim.corpora import Dictionary
from gensim.models import LdaModel

with open("../data/corpus_test.pkl", "rb") as f:
    corpus = pickle.load(f)

corpus_dictionary = Dictionary(corpus)
corpus = [corpus_dictionary.doc2bow(text) for text in corpus]

CORPUS = corpus
TOPIC_NUM = 10
lda = LdaModel(corpus, num_topics=TOPIC_NUM)

doc_topic_mat = lda.get_document_topics([(0,1), (1,1)])
term_topic_mat = lda.get_term_topics(1)
topic_term_mat = lda.get_topic_terms(1)

print("a")
