# -*- coding: utf-8 -*\
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from gensim import corpora, models
from gensim.matutils import sparse2full
from gensim.sklearn_api import ldamodel

import pandas as pd
import os

class GensimTfidfVectorizer(BaseEstimator, TransformerMixin):
    def __init__(self, dir_path='.', tofull = False, tokenizer = None, n_gram = 1):     
        self._lexicon_path = os.path.join(dir_path, 'corpus.dict')
        self._tfidf_path = os.path.join(dir_path, 'tfidf.model')
        self.lexicon = None
        self.tfidf = None
        self.tofull = tofull
        self.tokenizer = tokenizer
        self.texts = []
        self.n_gram = n_gram

        self.load()
    def load(self):
        if( os.path.exists(self._lexicon_path)):
            self.lexicon = corpora.Dictionary.load(self._lexicon_path)
        if(os.path.exists(self._tfidf_path)):
            self.tfidf = models.TfidfModel().load(self._tfidf_path)
    
    def save(self):
        self.lexicon.save(self._lexicon_path)
        self.tfidf.save(self._tfidf_path)

    def fit(self, documents, labels=None):
        self.lexicon = corpora.Dictionary(documents)
        self.tfidf = models.TfidfModel([
            self.lexicon.doc2bow(doc) for doc in documents],
            ld2word = self.lexicon)
        return self

    def transform(self, documents):
        def generator():
            for document in documents:
                vec = self.tfidf[self.lexicon.doc2bow(document)]
                if(self.tofull):
                    yield sparse2full(vec)
                else:
                    yield vec
        return list(generator())

    def fit_transform(self, documents, labels = None):
        if(self.n_gram == 1):
            self.texts = [self.tokenizer(s) for s in documents]
        else:
            for s in documents:
                tmp = self.tokenizer(s)
                self.texts.append([' '.join(tmp[i:i+self.n_gram]) for i in range(len(tmp)-self.n_gram +1)])

        self.lexicon = corpora.Dictionary(self.texts)
        self.tfidf = models.TfidfModel(corpus=[self.lexicon.doc2bow(doc) for doc in self.texts], id2word = self.lexicon)
        self.save()
        
        def generator():
            for document in self.texts:
                vec = self.tfidf[self.lexicon.doc2bow(document)]
                if(self.tofull):
                    yield sparse2full(vec)
                else:
                    yield vec
        return list(generator())

    def get_id2word(self):
        return self.lexicon