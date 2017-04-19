#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2017/3/30 10:25
# @Author  : ConanCui
# @Site    : 
# @File    : kmeansforuser.py
# @Software: PyCharm Community Edition

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn import metrics
import random


import WNMFclass
import HSRx2main

def nearest(cluster_centers_,X,n):
    '''
    :param cluster_centers_: 聚类中心坐标
    :param X:所有的样本数
    :return:返回每类最近邻的n个样本的id
    '''
    n_cluster = cluster_centers_.shape[0]
    m = X.shape[0]
    order = {}
    for i in range(n_cluster):
        center = cluster_centers_[i,:]
        distance = []

        for j in range(m):
            distance.append([j,np.sum((X[j,:] - center)**2)])

        distance.sort(key = lambda a : a[1])
        order[i] = (np.array(distance[0:n] , dtype= "int32")[:,0])
    return order

def Kmeans(U,n_clusters):
    kmeans = KMeans(n_clusters=n_clusters, precompute_distances=False, random_state=9)
    y_pred = kmeans.fit_predict(U)
    cluster_centers_ = kmeans.cluster_centers_
    return y_pred , cluster_centers_

def ShowNearest(fea,cluster_centers_,n_clusters,attribute_type,method_type):
    order = nearest(cluster_centers_, fea, sampleNumber)
    print("----------------------------------------------------------------------------------------------------------")
    for i in range(n_clusters):
        #所有符合第i+1类标准的
        print("The  %d cluster Method is %s || Attribute is %s" % (i + 1,method_type , attribute_type))
        content = df._values[order[i],:]

        Showmovie(content)
        print("----------------------------------------------------------------------------------------------------------")
def Showmovie(content):
    for info in content:
        output = info[0]
        if info[4] == 1:
            output = output + " |unknow| "
        if info[5] == 1:
            output = output + " |Action| "
        if info[6] == 1:
            output = output + " |Adventure| "
        if info[7] == 1:
            output = output + " |Animation| "
        if info[8] == 1:
            output = output + " |Children| "
        if info[9] == 1:
            output = output + " |Comedy| "
        if info[10] == 1:
            output = output + " |Crime| "
        if info[11] == 1:
            output = output + " |Documentary| "
        if info[12] == 1:
            output = output + " |Drama| "
        if info[13] == 1:
            output = output + " |Fantasy| "
        if info[14] == 1:
            output = output + " |Film-Noir| "
        if info[15] == 1:
            output = output + " |Horror| "
        if info[16] == 1:
            output = output + " |Musical| "
        if info[17] == 1:
            output = output + " |Mystery| "
        if info[18] == 1:
            output = output + " |Romance| "
        if info[19] == 1:
            output = output + " |Sci-Fi| "
        if info[20] == 1:
            output = output + " |Thriller| "
        if info[21] == 1:
            output = output + " |War| "
        if info[22] == 1:
            output = output + " |Western| "
        print output


def load_user_fea():
    # 提取原始特征
    n_cluster = {'year': 5 ,'genre': 5 }
    # n_cluster['age'] = (df.age.max()//10 - df.age.min()//10)
    n_movies = 1682
    YearFea = np.zeros((n_movies, 1))
    GenreFea = np.zeros((n_movies, 19))

    for line in df.itertuples():
        YearFea[line.Index -1 ] = float(line[1].split("(")[-1].split(")")[0])
        GenreFea[line.Index -1,: ] = line[5:]
    Fea = {'year': 2 ,'genre': 2 }
    Fea['year'] = YearFea
    Fea['genre'] = GenreFea
    return n_cluster,Fea

def load_ratings():
    # 录入评分数据
    header = ['user_id', 'item_id', 'rating', 'timestamp']
    cf = pd.read_csv('./ml-100k/ml-100k/u.data', sep='\t', names=header)
    n_users = cf.user_id.unique().shape[0]
    n_items = cf.item_id.unique().shape[0]
    # Create training matrix
    R = np.zeros((n_users, n_items))
    for line in cf.itertuples():
        R[line[1] - 1, line[2] - 1] = line[3]
    return R

def bulid_info(method,socre,centers):
    return info(method,socre,centers)

class info():
    def __init__(self,method,socre,centers):
        self.method = method
        self.socre = socre
        self.centers = centers

def NMIsocre(methodtype,Feamethod,toObject,attribute_type = None,):
    attribute = ['year', 'genre']
    socre = {}
    centers = {}
    for i in range(len(attribute)):
        index = attribute[i]
        gnd, cluster_centers_ = Kmeans(Fea[index],n_cluster[index])
        pre, cluster_centers_ = Kmeans(Feamethod, n_cluster[index])
        socre[index] = metrics.normalized_mutual_info_score(gnd, pre)
        centers[index] = cluster_centers_
    return toObject(methodtype,socre,centers)

def ShowSampleArrtibute(feamethod,socre,pair, n_cluster):
    attribute = ['year', 'genre']
    for i in attribute:
        method_type = pair[i]
        n = n_cluster[i]
        fea = feamethod[method_type]
        cluster_centers_ = socre[method_type].centers[i]
        ShowNearest(fea, cluster_centers_, n, i, method_type)

def get_fea(methodtype):
    feamethod = {}
    # U_WNMF, V = WNMFclass.WNMF(R, k=20, lamda=0)
    # # ALS-WR
    # # U_ALS , V = ALS_WR.ALS_WR()
    # # U_ALS = U_ALS.T
    # feamethod[methodtype[0]] = U_WNMF
    # # 基本的矩阵分解提取特征,一层，分解数为20
    # mfOneLayer = HSRx2main.HSRtest(n_epochs_nmf=150, n_epochs_wnmf=150, lamda_wnmf=0, gama=1, beta=1, type='linear')
    # mfOneLayer.Loaddata()
    # mfOneLayer.Setparamets(M=[20], N=[20], lamda=0, n_epochs=100, alpha=0.5)
    # mfOneLayer.Initialization()
    # mfOneLayer.Factorization()
    # feamethod[methodtype[1]] = mfOneLayer.U[1]
    # 深度矩阵分解提取特征,两层，分解数为(M=[20,100], N=[20,1000])
    mfTwoLayer = HSRx2main.HSRtest(n_epochs_nmf=100, n_epochs_wnmf=100, lamda_wnmf=0, gama=1, beta=1, type='linear')
    mfTwoLayer.Loaddata()
    mfTwoLayer.Setparamets(M=[20,100], N=[20,1000], lamda=0, n_epochs=100, alpha=0.5)
    mfTwoLayer.Initialization()
    mfTwoLayer.Factorization()
    feamethod[methodtype[0]] = mfTwoLayer.V_[1].T
    feamethod[methodtype[1]] = mfTwoLayer.V[1].T
    # feamethod[methodtype[4]] = U_ALS
    return feamethod

def bestmethodforattributr(methodtype, feamethod):
    socre = {}
    for i in range(len(methodtype)):
        socre[methodtype[i]] = NMIsocre(methodtype[i], feamethod[methodtype[i]], bulid_info)

    # data = np.array([socre['WNMF'].socre.values(), socre['HSR1'].socre.values(), socre['HSR2_1'].socre.values(),
    #                  socre['HSR2_2'].socre.values()])
    data = np.array([socre['HSR2_1'].socre.values(),socre['HSR2_2'].socre.values()])

    best = np.argmax(data, axis=0)
    return best, socre


socre = {}
sampleNumber = 5
header = [ 'name', 'year','nan', 'url', 'unknown','Action','Adventure','Animation','Children','Comedy','Crime','Documentary','Drama','Fantasy','Film-Noir','Horror','Musical','Mystery','Romance','Sci-Fi','Thriller','War','Western']
df = pd.read_csv('./ml-100k/ml-100k/u.item', sep='|', names=header)

n_cluster , Fea = load_user_fea()

# R = load_ratings()

# methodtype = ['WNMF','HSR1','HSR2_1','HSR2_2']
methodtype = ['HSR2_1','HSR2_2']
feamethod = get_fea(methodtype)
attribute = ['year', 'genre']
best,socre = bestmethodforattributr(methodtype,feamethod)
pair = {attribute[i] : methodtype[best[i]] for i in range(len(attribute)) }

ShowSampleArrtibute(feamethod,socre,pair, n_cluster)
















