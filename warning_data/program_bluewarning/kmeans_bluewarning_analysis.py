
from sklearn.feature_extraction.text import CountVectorizer,TfidfTransformer
from sklearn.cluster import KMeans
import jieba
from sklearn import metrics
import numpy as np
import pandas as pd
import os
os.chdir('D:\warning data') #path
def kmeans_blue_warning(test_file_name,train_file_name,out_file_name):
    n_clusters=10
    abs_score_list = []
    ScoreList = []
    CountDict = dict()
    df = pd.read_csv(test_file_name)#'output.csv'
    TestData = df['summary'].tolist()
    TestIndex= df['id'].tolist()
    TrainDf=pd.read_csv(train_file_name, encoding='GB2312')
    TrainData=TrainDf['SUMMARY'].tolist()
    for d in TestData:
        TrainData.insert(1000000, d)
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
    bin_start = len(TrainData) - len(TestData)
    ModelDict = {}
    Risk = []
    Summary = []
    Category = []
    Variance = []
    with open(out_file_name, 'w') as outfile:
        while bin_start < len(km.labels_):
            scores = ScoreList[bin_start]
            cdfscore = CountDict[abs(scores)]
            Risk.append(round(1 / (1.0001 - cdfscore), 0))
            Summary.append(TrainData[bin_start])
            Category.append(km.labels_[bin_start])
            Variance.append(scores)
            bin_start = bin_start+ 1
        ModelDict = {'id': TestIndex, 'summary': Summary, 'risk': Risk, 'category': Category, 'variance': Variance}
        frame = pd.DataFrame(ModelDict, columns=['id', 'summary', 'risk', 'category', 'variance'])
        frame.to_csv(outfile)
kmeans_blue_warning('output.csv','SystemResultTrain.csv','ModelResult.csv')