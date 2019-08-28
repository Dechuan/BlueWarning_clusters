from sklearn.feature_extraction.text import CountVectorizer,TfidfTransformer
from sklearn.cluster import KMeans
import jieba
from sklearn import metrics
import pandas as pd
#import os
def kmeans_blue_warning(in_df):
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
        #Summary.append(TrainData[bin_start])
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
    ModelDict = {'id': TrainIndex,'risk': Risk,'category':Category}
    out_df= pd.DataFrame(ModelDict, columns=['id', 'risk','category'])
    return out_df