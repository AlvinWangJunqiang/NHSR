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
from sklearn.cluster import KMeans
from sklearn import metrics
import random


import WNMFclass
import NMFclass
import ALS_WR
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
        print content
        print("----------------------------------------------------------------------------------------------------------")

def load_user_fea():
    # 提取原始特征
    n_cluster = {'gen': 2 ,'age': 3 , 'occ' : 21 , 'zip' : 10}
    # n_cluster['age'] = (df.age.max()//10 - df.age.min()//10)
    n_users = df.id.unique().shape[0]
    AgeFea = np.zeros((n_users, 1))
    GenFea = np.zeros((n_users, n_cluster['gen']))
    OccFea = np.zeros((n_users, n_cluster['occ']))
    ZipFea = np.zeros((n_users, n_cluster['zip']))
    for line in df.itertuples():
        AgeFea[line[1] - 1] = line[2] // 10

        if line[3] == 'M':
            GenFea[line[1] - 1,0] = 1
        else:
            GenFea[line[1] - 1,1] = 1

        if line[4] == 'administrator':
            OccFea[line[1] - 1, 0] = 1
        elif line[4] == 'artist':
            OccFea[line[1] - 1, 1] = 1
        elif line[4] == 'doctor':
            OccFea[line[1] - 1, 2] = 1
        elif line[4] == 'educator':
            OccFea[line[1] - 1, 3] = 1
        elif line[4] == 'engineer':
            OccFea[line[1] - 1, 4] = 1
        elif line[4] == 'entertainment':
            OccFea[line[1] - 1, 5] = 1
        elif line[4] == 'executive':
            OccFea[line[1] - 1, 6] = 1
        elif line[4] == 'healthcare':
            OccFea[line[1] - 1, 7] = 1
        elif line[4] == 'homemaker':
            OccFea[line[1] - 1, 8] = 1
        elif line[4] == 'lawyer':
            OccFea[line[1] - 1, 9] = 1
        elif line[4] == 'librarian':
            OccFea[line[1] - 1, 10] = 1
        elif line[4] == 'marketing':
            OccFea[line[1] - 1, 11] = 1
        elif line[4] == 'none':
            OccFea[line[1] - 1, 12] = 1
        elif line[4] == 'other':
            OccFea[line[1] - 1, 13] = 1
        elif line[4] == 'programmer':
            OccFea[line[1] - 1, 14] = 1
        elif line[4] == 'retired':
            OccFea[line[1] - 1, 15] = 1
        elif line[4] == 'salesman':
            OccFea[line[1] - 1, 16] = 1
        elif line[4] == 'scientist':
            OccFea[line[1] - 1, 17] = 1
        elif line[4] == 'student':
            OccFea[line[1] - 1, 18] = 1
        elif line[4] == 'technician':
            OccFea[line[1] - 1, 19] = 1
        elif line[4] == 'writer':
            OccFea[line[1] - 1, 20] = 1

        if str.isdigit(line[5]) :
            index = int(line[5]) //10000
            ZipFea[line[1] - 1 ,index ] = 1
        else:
            ZipFea[line[1] - 1 ,1 ] = 1
    Fea = {'gen': 2 ,'age': 2 , 'occ' : 21 , 'zip' : 10}
    Fea['age'] = AgeFea
    Fea['gen'] = GenFea
    Fea['occ'] = OccFea
    Fea['zip'] = ZipFea
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
    attribute = ['age','gen','occ','zip']
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
    attribute = ['age', 'gen', 'occ', 'zip']
    for i in attribute:
        method_type = pair[i]
        n = n_cluster[i]
        fea = feamethod[method_type]
        cluster_centers_ = socre[method_type].centers[i]
        ShowNearest(fea, cluster_centers_, n, i, method_type)




def get_fea(methodtype):
    feamethod = {}
    U_WNMF, V = WNMFclass.WNMF(R, k=20, lamda=0)
    # ALS-WR
    # U_ALS , V = ALS_WR.ALS_WR()
    # U_ALS = U_ALS.T
    feamethod[methodtype[0]] = U_WNMF
    # 基本的矩阵分解提取特征,一层，分解数为20
    mfOneLayer = HSRx2main.HSRtest(n_epochs_nmf=150, n_epochs_wnmf=150, lamda_wnmf=8, gama=1, beta=1, type='linear')
    mfOneLayer.Loaddata()
    mfOneLayer.Setparamets(M=[20], N=[20], lamda=0, n_epochs=100, alpha=0.5)
    mfOneLayer.Initialization()
    mfOneLayer.Factorization()
    feamethod[methodtype[1]] = mfOneLayer.U[1]
    # 深度矩阵分解提取特征,两层，分解数为(M=[20,100], N=[20,1000])
    mfTwoLayer = HSRx2main.HSRtest(n_epochs_nmf=100, n_epochs_wnmf=100, lamda_wnmf=8, gama=1, beta=1, type='linear')
    mfTwoLayer.Loaddata()
    mfTwoLayer.Setparamets(M=[20,100], N=[20,1000], lamda=0, n_epochs=100, alpha=0.5)
    mfTwoLayer.Initialization()
    mfTwoLayer.Factorization()
    feamethod[methodtype[2]] = mfTwoLayer.U_[1]
    feamethod[methodtype[3]] = mfTwoLayer.U[1]
    # feamethod[methodtype[4]] = U_ALS
    return feamethod

def bestmethodforattributr(methodtype, feamethod):
    socre = {}
    for i in range(len(methodtype)):
        socre[methodtype[i]] = NMIsocre(methodtype[i], feamethod[methodtype[i]], bulid_info)

    data = np.array([socre['WNMF'].socre.values(), socre['HSR1'].socre.values(), socre['HSR2_1'].socre.values(),
                     socre['HSR2_2'].socre.values()])

    best = np.argmax(data, axis=0)
    return best, socre


socre = {}
sampleNumber = 5
header = ['id', 'age', 'gen', 'occ', 'zip']
df = pd.read_csv('./ml-100k/ml-100k/u.user', sep='|', names=header)
n_cluster , Fea = load_user_fea()

R = load_ratings()

methodtype = ['WNMF','HSR1','HSR2_1','HSR2_2']
feamethod = get_fea(methodtype)
attribute = ['age', 'gen', 'occ', 'zip']
best,socre = bestmethodforattributr(methodtype,feamethod)
pair = {attribute[i] : methodtype[best[i]] for i in range(len(attribute)) }

ShowSampleArrtibute(feamethod,socre,pair, n_cluster)

A = 1














