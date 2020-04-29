# -*- coding: utf-8 -*
import MeCab
import os
import pandas as pd
import numpy as np
from pprint import pprint

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler
from gensim import corpora, models
from MyGensim import GensimTfidfVectorizer

import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.font_manager as fm

def searchFiles(path):
    filelist = []
    filenames = os.listdir(path)
    for filename in filenames:
        file_path = os.path.join(path, filename)
        filelist.append(file_path)
    return filelist

def getNouns(text):
    tokenizer = MeCab.Tagger()
    parsed = tokenizer.parse(text)
    word_tag = [w for w in parsed.split("\n")]
    pos = []
    for word_ in word_tag[:-2]:
        word = word_.split("\t")
        tag = word[1].split(",")
        if(len(word[0])> 1 and 'NNG' in tag[0] or 'NNP' in tag[0]):
            pos.append(word[0])
    return pos

def getNVM_lemma(text):
    tokenizer = MeCab.Tagger()
    parsed = tokenizer.parse(text)
    word_tag = [w for w in parsed.split("\n")]
    pos = []
    tags = ['NNG','NNP','VV','VA','VX','VCP','VCN']

    for word_ in word_tag[:-2]:
        word = word_.split("\t")
        tag = word[1].split(",")
        if(len(word[0]) < 2):
            continue
        if(tag[-1] != '*'):
            t = tag[-1].split('/')
            if(len(t[0]) > 1 and ('VV' in t[1] or 'VA' in t[1] or 'VX' in t[1])):
                pos.append(t[0])
        else:
            if(tag[0] in tags):
                pos.append(word[0])
    return pos

def main():
    reviews = []
    for filePath in  searchFiles('./Reviews/IP/'):
        review = pd.read_csv(filePath, encoding = 'utf-8')
        reviews.append(review)

    docs = pd.concat(reviews, ignore_index=True)
    docs['내용'] = docs.apply(lambda x: x['내용']*int(np.log2(2 + x['공감수'])), axis = 1)
    print('리뷰 읽기 끝')

    vect = GensimTfidfVectorizer(tokenizer=getNVM_lemma, n_gram = 2, dir_path='.')
    
    texts = vect.fit_transform(docs['내용'])
    id2word = vect.get_id2word()
    data = vect.texts

    print('벡터화 끝')  
    lda = models.LdaModel(corpus=texts, 
                        id2word=id2word, 
                        num_topics=20, 
                        update_every=1, 
                        chunksize=1000, 
                        passes=10,
                        alpha='auto',
                        eta='auto',
                        per_word_topics=False)

    topics = sorted(lda.show_topics(num_topics = 20, num_words=20, formatted=False), key=lambda x:x[0])
    
    pprint(topics)
    print('')

    for row in lda[texts][2]:
        pprint(row)

    print(lda.log_perplexity(texts))
    cm = models.CoherenceModel(model=lda, texts = data, dictionary=id2word, coherence = 'c_v')
    print(cm.get_coherence())

    return None

if __name__=='__main__':
    main()