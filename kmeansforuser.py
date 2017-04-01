#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2017/3/30 10:25
# @Author  : ConanCui
# @Site    : 
# @File    : kmeans.py
# @Software: PyCharm Community Edition

import numpy as np
import pandas as pd
from sklearn import cross_validation as cv
from sklearn import preprocessing
import random

import HSRx2main
from sklearn.cluster import KMeans
from sklearn import metrics

def Genre(content):
    genre = " "
    if content[5] == 1:
        genre = genre + " | unknow "
    if content[6] == 1:
        genre = genre + " | Action "
    if content[7] == 1:
        genre = genre + " | Adventure "
    if content[8] == 1:
        genre = genre + " | Animation "
    if content[9] == 1:
        genre = genre + " | Childrens "
    if content[10] == 1:
        genre = genre + " | Comedy "
    if content[11] == 1:
        genre = genre + " | Crime "
    if content[12] == 1:
        genre = genre + " | Documentary "
    if content[13] == 1:
        genre = genre + " | Drama "
    if content[14] == 1:
        genre = genre + " | Fantasy "
    if content[15] == 1:
        genre = genre + " | Film_Noir "
    if content[16] == 1:
        genre = genre + " | Horror "
    if content[17] == 1:
        genre = genre + " | Musical "
    if content[18] == 1:
        genre = genre + " | Mystery "
    if content[19] == 1:
        genre = genre + " | Romance  "
    if content[20] == 1:
        genre = genre + " | Sci-Fi "
    if content[21] == 1:
        genre = genre + " | Thriller "
    if content[22] == 1:
        genre = genre + " | War "
    if content[23] == 1:
        genre = genre + " | Western "
    return genre+"|"

def AnalysisMovies(preOneLayer):

    for i in range(classNumber):
        #所有符合第i+1类标准的
        index = np.array(np.where(preOneLayer == i))[0]
        print("----------------------------------------------------------------------------------------------------------")
        print("The number of %d cluster is %d" % (i + 1, len(index)))
        #随机抽取五个
        for temp in range(sampleNumber):
            temp = random.randint(0, len(index)-1)
            index_temp = index[temp]
            #从df的第index_temp进行抽取，作为第i+1类的例子
            content = df._values[index_temp,:]
            print str(content[1:3]) + Genre(content)
        print("----------------------------------------------------------------------------------------------------------")

#参数设置
sampleNumber = 5
classNumber = 5

#录入数据
header = ['movie_id', 'movie title', 'release date', 'video release date','IMDb URL','unknown ','Action ', 'Adventure' , 'Animation '
              ,'Childrens ',' Comedy ',' Crime','Documentary', 'Drama',' Fantasy ',
              'Film-Noir ', 'Horror ',' Musical', ' Mystery ', 'Romance', ' Sci-Fi',
              'Thriller', ' War', ' Western ']
df = pd.read_csv('./ml-100k/ml-100k/u.item', sep='|', names=header)

# 提取原始特征(以genre来作为特征)
n_items = df.movie_id.unique().shape[0]
featureGenre = np.zeros((n_items,19))
for line in df.itertuples():
    featureGenre[line[1] - 1, :] = line[6:]



# 基本的矩阵分解提取特征,一层，分解数为20
mfOneLayer = HSRx2main.HSR(n_epochs_nmf=100, n_epochs_wnmf=100, lamda_wnmf=0, gama=1, beta=1, type='linear')
mfOneLayer.Loaddata(test_size=0.001)
mfOneLayer.Setparamets(M=[20], N=[20], lamda=0, n_epochs=100, alpha=0.5)
mfOneLayer.Initialization()
mfOneLayer.Factorization()
trainOneLayer, testOneLayer = mfOneLayer.Monitor(save = False ,show = False)


# 深度矩阵分解提取特征,两层，分解数为
mfTwoLayer = HSRx2main.HSR(n_epochs_nmf=100, n_epochs_wnmf=100, lamda_wnmf=0, gama=1, beta=1, type='linear')
mfTwoLayer.Loaddata(test_size=0.001)
mfTwoLayer.Setparamets(M=[20,100], N=[20,1000], lamda=0, n_epochs=100, alpha=0.5)
mfTwoLayer.Initialization()
mfTwoLayer.Factorization()
trainTwoLayer, testTwoLayer = mfTwoLayer.Monitor(save = False ,show = False)

#用原始特征做分类，将分类的结果作为真实标签
gnd = KMeans(n_clusters=classNumber, precompute_distances=False ).fit_predict(featureGenre)
print "电影流派分类结果展示"
AnalysisMovies(gnd)

#用一层分解中的V做分类，将分类的结果作为预测标签preOneLayer
preOneLayer = KMeans(n_clusters = classNumber, precompute_distances=False).fit_predict(mfOneLayer.V[1].T)
print "OneLayer的分类结果展示"
AnalysisMovies(preOneLayer)

#用两层分解中的第一层分解得到的V_[1]和第二层分解得到的V[1]做分类，将分类的结果作为预测标签preTwoLayer1，preTwoLayer2
preTwoLayer1 = KMeans(n_clusters = classNumber, precompute_distances=False).fit_predict(mfOneLayer.V_[1].T)
preTwoLayer2 = KMeans(n_clusters = classNumber, precompute_distances=False).fit_predict(mfOneLayer.V[1].T)
print "preTwoLayer1的分类结果展示"
AnalysisMovies(preTwoLayer1)
print "preTwoLayer2的分类结果展示"
AnalysisMovies(preTwoLayer2)


#对比不同特征向量提取出的NMI分数
socre = metrics.normalized_mutual_info_score(gnd, gnd)
scoreOneLayer = metrics.normalized_mutual_info_score(gnd, preOneLayer)
scoreTwoLayer1 = metrics.normalized_mutual_info_score(gnd, preTwoLayer1)
scoreTwoLayer2 = metrics.normalized_mutual_info_score(gnd, preTwoLayer2)
print("----------------------------------------------------------------------------------------------------------")
print ("真实标签与真实标签之间的NMI分数为：%f" %socre)

print("----------------------------------------------------------------------------------------------------------")
print ("基本矩阵分解的分类结果与真实标签之间的NMI分数为：%f" %scoreOneLayer)

print("----------------------------------------------------------------------------------------------------------")
print ("两层分解中第一层与真实标签之间的NMI分数为：%f" %scoreTwoLayer1)
print ("两层分解中第二层与真实标签之间的NMI分数为：%f" %scoreTwoLayer2)
print("----------------------------------------------------------------------------------------------------------")

