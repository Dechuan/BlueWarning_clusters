#!/usr/bin/python
# coding=utf-8
# 采用TF-IDF方法提取文本关键词
# http://scikit-learn.org/stable/modules/feature_extraction.html#tfidf-term-weighting
import sys,codecs
import pandas as pd
import numpy as np
import jieba.posseg
import jieba.analyse
from sklearn import feature_extraction
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import CountVectorizer,TfidfTransformer
from sklearn.cluster import KMeans
import jieba
import re
from sklearn import metrics
import pandas as pd
"""
       TF-IDF权重：
           1、CountVectorizer 构建词频矩阵
           2、TfidfTransformer 构建tfidf权值计算
           3、文本的关键字
           4、对应的tfidf矩阵
"""
# 数据预处理操作：分词，去停用词，词性筛选
def dataPrepos(text):
    l = []
    pos = ['n', 'nz', 'v', 'vd', 'vn', 'l', 'a', 'd']  # 定义选取的词性
    text=re.sub(r'\b\d+(?:\.\d+)?\s+', '', text)
    seg = jieba.posseg.cut(text)  # 分词
    for i in seg:
        l.append(i.word)

    return l

# tf-idf获取文本top10关键词
def getKeywords_tfidf(data,topK):
    idList, summaryList = data['id'], data['summary']
    corpus = [] # 将所有文档输出到一个list中，一行就是一个文档
    for index in range(len(idList)):
        text = summaryList[index]# 读取summary
        text = dataPrepos(text) # 文本预处理
        text = " ".join(text) # 连接成字符串，空格分隔
        corpus.append(text)

    # 1、构建词频矩阵，将文本中的词语转换成词频矩阵
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(corpus) # 词频矩阵,a[i][j]:表示j词在第i个文本中的词频
    # 2、统计每个词的tf-idf权值
    transformer = TfidfTransformer()
    tfidf = transformer.fit_transform(X)
    # 3、获取词袋模型中的关键词
    word = vectorizer.get_feature_names()
    # 4、获取tf-idf矩阵，a[i][j]表示j词在i篇文本中的tf-idf权重
    weight = tfidf.toarray()
    # 5、打印词语权重
    ids, summary, keys = [], [], []
    for i in range(len(weight)):
        #print(u"-------这里输出第", i+1 , u"篇文本的词语tf-idf------")
        ids.append(idList[i])
        summary.append(summaryList[i])
        df_word,df_weight = [],[] # 当前文章的所有词汇列表、词汇对应权重列表
        for j in range(len(word)):
           #print(word[j],weight[i][j])
            df_word.append(word[j])
            df_weight.append(weight[i][j])
        df_word = pd.DataFrame(df_word,columns=['word'])
        df_weight = pd.DataFrame(df_weight,columns=['weight'])
        word_weight = pd.concat([df_word, df_weight], axis=1) # 拼接词汇列表和权重列表
        word_weight = word_weight.sort_values(by="weight",ascending = False) # 按照权重值降序排列
        keyword = np.array(word_weight['word']) # 选择词汇列并转成数组格式
        word_split = [keyword[x] for x in range(0,topK)] # 抽取前topK个词汇作为关键词
        word_split = " ".join(word_split)
        keys.append(word_split)

    #result = pd.DataFrame({"id": ids, 'summary':summary, "key": keys},columns=['id','summary','key'])
    return keys
def kmeans_blue_warning(in_df,keys):
    n_clusters=10
    abs_score_list = []
    ScoreList = []
    CountDict = dict()
    TrainData = in_df['summary'].tolist()
    TrainIndex= in_df['id'].tolist()
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform([" ".join([b for b in jieba.cut(a)]) for a in TrainData])
    word = vectorizer.get_feature_names()
    transformer = TfidfTransformer()
    tfidf = transformer.fit_transform(X.toarray())
    km=KMeans(10)
    km.fit(tfidf.toarray())
    for i in tfidf.toarray():
        score = km.score(i.reshape(1, -1))
        ScoreList.append(score)
        abs_score_list.append(abs(score))
    sort_score_list = sorted(abs_score_list)
    scorelen = len(sort_score_list)
    for i in sort_score_list:
        count = 0
        for j in sort_score_list:
            if j <= i:
                count = count + 1
        CountDict[i] = count / scorelen
    bin_start=0
    ModelDict = {}
    Risk = []
    Summary = []
    Category = []
    Variance = []
    i=0
    while bin_start < len(km.labels_):
        scores = ScoreList[bin_start]
        cdfscore = CountDict[abs(scores)]
        Risk.append(round(1 / (1.0001 - cdfscore), 0))
        Summary.append(TrainData[bin_start])
        Category.append(km.labels_[bin_start])
        #Variance.append(scores)
        bin_start = bin_start+ 1
    while i<len(Risk):
        if Risk[i]>=1000:
            Risk[i]='1000+'
        elif (Risk[i]<1000)&(Risk[i]>=100):
            Risk[i]='100+'
        elif (Risk[i]<100)&(Risk[i]>=10):
            Risk[i]='10+'
        elif Risk[i]<10:
            Risk[i]='1+'
        i=i+1

    ModelDict = {'id': TrainIndex,'summary':Summary,'risk': Risk,'category':Category,'keys':keys}
    out_df= pd.DataFrame(ModelDict, columns=['id', 'summary','risk','category','keys'])
    return out_df

def main():
    # 读取数据集
    dataFile = 'data/90days.csv'
    data = pd.read_csv(dataFile,encoding="gb2312")
    # 停用词表
   # stopkey = [w.strip() for w in codecs.open('data/stopWord.txt', 'r',encoding='gb18030').readlines()]
    # tf-idf关键词抽取
    keys = getKeywords_tfidf(data, 10)
    result = kmeans_blue_warning(data,keys)
    result.to_csv("result/90days_result.csv",index=False)

if __name__ == '__main__':
    main()
