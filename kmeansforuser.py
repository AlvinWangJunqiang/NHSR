#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2017/3/30 10:25
# @Author  : ConanCui
# @Site    : 
# @File    : kmeansforuser.py
# @Software: PyCharm Community Edition

import numpy as np
import pandas as pd
from sklearn import cross_validation as cv
from sklearn import preprocessing
from sklearn.cluster import KMeans
from sklearn import metrics
import random


import WNMFclass
import HSRx2main

def nearest(cluster_centers_,X,n):
    '''
    :param cluster_centers_: 聚类中心坐标
    :param X:样本的特征矩阵，每一行代表一个样本，每一列代表一种特征
    :return:返回每类最近邻（离聚类中心最近）的n个样本的id
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
    '''

    :param U: 矩阵分解得到的特征矩阵
    :param n_clusters: 聚类的数目
    :return: 预测的标签，聚类的中心坐标
    '''
    kmeans = KMeans(n_clusters=n_clusters, precompute_distances=False, random_state=9)
    y_pred = kmeans.fit_predict(U)
    cluster_centers_ = kmeans.cluster_centers_
    return y_pred , cluster_centers_

def ShowNearest(fea,cluster_centers_,n_clusters,attribute_type,method_type):
    '''

    :param fea: 矩阵分解得到的特征矩阵
    :param cluster_centers_: 聚类中心坐标
    :param n_clusters: 聚类中心数量
    :param attribute_type:
    :param method_type:
    :return:
    '''
    order = nearest(cluster_centers_, fea, sampleNumber)
    print("----------------------------------------------------------------------------------------------------------")
    for i in range(n_clusters):
        #所有符合第i+1类标准的
        print("The  %d cluster Method is %s || Attribute is %s" % (i + 1,method_type , attribute_type))
        content = df._values[order[i],:]
        print content
        print("----------------------------------------------------------------------------------------------------------")

def load_user_fea():
    '''

    :return: 每一个类别的聚类数目，从是原始数据中获取的特征
    '''
    # 提取原始特征
    n_cluster = {'gen': 2 ,'age': 3 , 'occ' : 5 , 'zip' : 5}
    # n_cluster['age'] = (df.age.max()//10 - df.age.min()//10)
    n_users = df.id.unique().shape[0]
    AgeFea = np.zeros((n_users, 1))
    GenFea = np.zeros((n_users, 2))
    OccFea = np.zeros((n_users, 21))
    ZipFea = np.zeros((n_users, 10))
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
    '''

    :param methodtype:
    :param Feamethod:
    :param toObject:
    :param attribute_type:
    :return:#   返回为每一种方法，对应不同属性的分数，以及对应不同属性的聚类中心
    '''
    attribute = ['age','gen','occ','zip']
    socre = {}
    centers = {}
    for i in range(len(attribute)):
        index = attribute[i]
        # 这里的Fea和Feamethod的用户id应该相同，因为Fea是从u.user中读取
        # 出来的，顺序为1-。。。。，而Feamethod是从u.data中读取的，第i行代表第i个用户
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
    '''

    :param methodtype:
    :return:不同方法所提取出来的用户特征
    '''
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
    feamethod[methodtype[0]] = mfTwoLayer.U_[1]
    feamethod[methodtype[1]] = mfTwoLayer.U[1]
    # feamethod[methodtype[4]] = U_ALS
    return feamethod

def bestmethodforattributr(methodtype, feamethod):
    '''

    :param methodtype:
    :param feamethod:
    :return:data中存储了
    '''
    socre = {}
    for i in range(len(methodtype)):
        socre[methodtype[i]] = NMIsocre(methodtype[i], feamethod[methodtype[i]], bulid_info)
    # socre中包含了不同的方法对应的不同属性的分数，以及对应不同属性的聚类中心
    # data = np.array([socre['WNMF'].socre.values(), socre['HSR1'].socre.values(), socre['HSR2_1'].socre.values(),
    #                  socre['HSR2_2'].socre.values()])
    data = np.array([socre['HSR2_1'].socre.values(),socre['HSR2_2'].socre.values()])

    best = np.argmax(data, axis=0)
    return best, socre

def ShowSample(Fea,cluster,sampleNumber,method_type):
    '''
    不考虑NMI分数，也不考虑属性，直接对每个分解得到的因子进行分解
    :param feamethod: 方法提取出的特征
    :param cluster: 要聚成多少类别
    :param sampleNumber: 每类别抽出多少个最近邻的
    :return:
    '''
    gnd, cluster_centers_ = Kmeans(Fea, cluster)
    order = nearest(cluster_centers_, Fea, sampleNumber)
    print("----------------------------------------------------------------------------------------------------------")
    for i in range(cluster):
        #所有符合第i+1类标准的
        print("The  %d cluster Method is %s " % (i + 1,method_type ))
        content = df._values[order[i],:]
        print content
        print("----------------------------------------------------------------------------------------------------------")

socre = {}
sampleNumber = 5
header = ['id', 'age', 'gen', 'occ', 'zip']
df = pd.read_csv('./ml-100k/ml-100k/u.user', sep='|', names=header)
n_cluster , Fea = load_user_fea()

R = load_ratings()

# methodtype = ['WNMF','HSR1','HSR2_1','HSR2_2']
methodtype = ['HSR2_1','HSR2_2']
feamethod = get_fea(methodtype)

# attribute = ['age', 'gen', 'occ', 'zip']
# best,socre = bestmethodforattributr(methodtype,feamethod)
# pair = {attribute[i] : methodtype[best[i]] for i in range(len(attribute)) }
# ShowSampleArrtibute(feamethod,socre,pair, n_cluster)

ShowSample(Fea = feamethod['HSR2_1'],sampleNumber=5,cluster=5,method_type='HSR2_1')
ShowSample(Fea = feamethod['HSR2_2'],sampleNumber=5,cluster=5,method_type='HSR2_2')
















