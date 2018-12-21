#-*- coding: utf-8 -*-
import codecs
import pickle


def read_data(filename):
    with codecs.open(filename, encoding='utf-8', mode='r') as f:
        data = [line.split('\t') for line in f.read().splitlines()]
        data = data[1:] #헤더 제외
    return data

# corpus = read_data('../data/ratings_train.txt')
corpus = read_data('../data/ratings_train.txt')

from konlpy.tag import Twitter
tagger = Twitter()

def tokenize(doc):
    return ['/'.join(t) for t in tagger.pos(doc, norm=True, stem=True)]

train_docs = [row[1] for row in corpus]
sentences = [tokenize(d) for d in train_docs]
new_sentences = []

for l in sentences:
    new_l = []
    for w in l:
        new_l.append(w.encode("utf-8", "ignore"))

    new_sentences.append(new_l)

with open("../data/corpus_train.pkl", "wb") as f:
    pickle.dump(new_sentences, f)
