#-*- coding: utf-8 -*-
import itertools

from gensim.models.word2vec import Text8Corpus

sentences = list(itertools.islice(Text8Corpus('/path/to/text8'),None))
