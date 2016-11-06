# -*- coding: utf-8 -*-
# @Time    : 16/11/4 下午6:02
# @Author  : liulei
# @Brief   : 
# @File    : lda.py
# @Software: PyCharm Community Edition
import re
import time
from operator import add
import os, sys
import pandas as pd
from pyspark.mllib.feature import HashingTF
from pyspark.mllib.feature import IDF
import cPickle as pickle
import numpy as np
import jieba
import jieba.analyse
import jieba.posseg as pseg
from pyspark.mllib.clustering import LDA, LDAModel
from pyspark.mllib.linalg import Vectors
from datetime import datetime as dt

from pyspark import SparkConf, SparkContext

if __name__ == "__main__":
    sc = SparkContext(appName="SparkLDA")
    '''
    # jieba 設定斷詞
    jieba.set_dictionary('jieba/dict.txt.big')
    # jieba 設定停止詞
    jieba.analyse.set_stop_words("jieba/stop_words.txt")
    hashingTF = HashingTF()
    data = [""]
    documents = sc.textFile("data/testcsv.csv")
    for line in documents.collect():
        psegCut = pseg.cut(line)
        words = []
        for word, flag in psegCut:
            if (flag == "n"):
                words.append(word)
        data.append(list(words))

    data.remove("")
    documents = sc.parallelize(data)
    '''

    hashingTF = HashingTF()
    documents = sc.textFile("data/train.dat")


    def hashing(x):
        return hashingTF.transform([x]).indices[0]


    hashed = documents.flatMap(lambda line: line).map(lambda word: (hashing(word), word)).distinct()
    hashed_word = pd.DataFrame(hashed.collect(),
                               columns=['hash', 'word']).set_index('hash')
    # hashingTF = HashingTF()
    # Tf-Idf的生成
    tf = hashingTF.transform(documents)
    tf.cache()
    idf = IDF().fit(tf)
    tf_idf_data = idf.transform(tf)
    print dt.now().strftime('%Y/%m/%d %H:%M:%S')
    K = 5

    # Index documents with unique IDs
    corpus_data = tf_idf_data.zipWithIndex().map(lambda x: [x[1], x[0]]).cache()
    print corpus_data
    # Cluster the documents into three topics using LDA
    ldaModel = LDA.train(corpus_data, k=K)

    # Output topics. Each is a distribution over words (matching word count vectors)
    print "Learned topics (as distributions over vocab of " + str(
        ldaModel.vocabSize()) + " words):"
    topics = ldaModel.topicsMatrix()
    print dt.now().strftime('%Y/%m/%d %H:%M:%S')


    def idx_to_word(idx):
        res = hashed_word.ix[idx].word
        if type(res) == pd.Series:
            return res.to_dict().values()[0]
        else:
            return res


    rep_num = 20

    for topic in range(K):
        print "Topic " + str(topic) + ":"
        temp_w = []
        temp_t = []
        for word in range(0, ldaModel.vocabSize()):
            top = topics[word][topic]
            if (top != 0):
                temp_w.append(word)
                temp_t.append(top)
        temp_w = np.array(temp_w)

        temp_t = np.array(temp_t)
        idx = np.argsort(temp_t)[::-1]
        print u','.join(map(idx_to_word, temp_w[idx[:20]])).encode(
            'utf-8').strip()
        print temp_t[idx[:20]]
    sc.stop()
