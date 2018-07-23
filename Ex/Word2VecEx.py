#-*- coding: utf-8 -*-
import pickle


with open("../data/corpus_training.pkl", "rb") as f:
    sentences = pickle.load(f)

from gensim.models import Word2Vec
model = Word2Vec(sentences)
model.init_sims(replace=True)


from konlpy.tag import Twitter
tagger = Twitter()

def tokenize(doc):
    return ['/'.join(t) for t in tagger.pos(doc, norm=True, stem=True)]

print(u'눈물 / 감성 : ', model.similarity(*tokenize(u'눈물 감성')))
print(u'액션 / 연출 : ', model.similarity(*tokenize(u'액션 연출')))


from konlpy.utils import pprint
pprint(model.most_similar(positive=tokenize(u'여배우 남자'), negative=tokenize(u'배우'), topn=1))