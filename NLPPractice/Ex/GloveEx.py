import pickle

from glove import Corpus, Glove

with open("../data/corpus_training.pkl", "rb") as f:
    corpus = pickle.load(f)


glove_corpus = Corpus()
glove_corpus.fit(corpus, window=5)

model = Glove()
model.fit(glove_corpus.matrix, epochs=10, no_threads=4)
model.add_dictionary(glove_corpus.dictionary)
model.save("../model/model_glove")

model.most_similar("영화/Noun")
