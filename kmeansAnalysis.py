#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2017/5/7 10:35
# @Author  : ConanCui
# @Site    : 
# @File    : kmeansAnalysis.py
# @Software: PyCharm Community Edition
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

import HSRx2main
def nearest(cluster_centers_,X,n):
    '''
    根据聚类中心，返回每个类别按照聚类中心远近的排序
    :param cluster_centers_: 聚类中心坐标
    :param X:样本的特征矩阵，每一行代表一个样本，每一列代表一种特征
    :return:返回每类最近邻（离聚类中心最近）的n个样本的id，即类别1，对应n个样本id
    类别2，对应n个样本id
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
    对Kmeans的重新包装
    :param U: 矩阵分解得到的特征矩阵
    :param n_clusters: 聚类的数目
    :return: 预测的标签，聚类的中心坐标
    '''
    kmeans = KMeans(n_clusters=n_clusters, precompute_distances=False, random_state=9)
    y_pred = kmeans.fit_predict(U)
    cluster_centers_ = kmeans.cluster_centers_
    return y_pred , cluster_centers_
def loadRatings():
    # 录入评分数据，用于进行深度矩阵分解
    header = ['user_id', 'item_id', 'rating', 'timestamp']
    cf = pd.read_csv('./ml-100k/ml-100k/u.data', sep='\t', names=header)
    n_users = cf.user_id.unique().shape[0]
    n_items = cf.item_id.unique().shape[0]
    # Create training matrix
    R = np.zeros((n_users, n_items))
    for line in cf.itertuples():
        R[line[1] - 1, line[2] - 1] = line[3]
    return R
def getFea(methodtype):
    '''
    :param methodtype:
    :return:不同方法所提取出来的用户特征
    '''
    feaOfUser = {}
    feaOfItem = {}
    mfTwoLayer = HSRx2main.HSRtest(n_epochs_nmf=100, n_epochs_wnmf=100, lamda_wnmf=0, gama=1, beta=1, type='linear')
    mfTwoLayer.Loaddata()
    mfTwoLayer.Setparamets(M=[20,100], N=[20,1000], lamda=0, n_epochs=100, alpha=0.5)
    mfTwoLayer.Initialization()
    mfTwoLayer.Factorization()
    feaOfUser[methodtype[0]] = mfTwoLayer.U_[1]
    feaOfUser[methodtype[1]] = mfTwoLayer.U[1]

    feaOfItem[methodtype[0]] = mfTwoLayer.V_[1].T
    feaOfItem[methodtype[1]] = mfTwoLayer.V[1].T

    return feaOfUser,feaOfItem
def ShowUser(Fea,cluster,sampleNumber,method_type):
    '''
    :param feamethod: 方法提取出的特征
    :param cluster: 要聚成多少类别
    :param sampleNumber: 每类别抽出多少个最近邻的用户
    :return:
    '''
    gnd, cluster_centers_ = Kmeans(Fea, cluster)
    order = nearest(cluster_centers_, Fea, sampleNumber)
    print("----------------------------------------------------------------------------------------------------------")
    for i in range(cluster):
        #所有符合第i+1类标准的
        print("The  %d cluster Method is %s " % (i + 1,method_type ))
        content = dfUser._values[order[i],:]
        print content
        print("----------------------------------------------------------------------------------------------------------")
def ShowItem(Fea,cluster,sampleNumber,method_type):
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
        content = dfItem._values[order[i],:]
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
        print("----------------------------------------------------------------------------------------------------------")

methodtype = ['HSR2_1','HSR2_2']
headerUser = ['id', 'age', 'gen', 'occ', 'zip']
dfUser = pd.read_csv('./ml-100k/ml-100k/u.user', sep='|', names=headerUser)
headerItem = [ 'name', 'year','nan', 'url', 'unknown','Action','Adventure','Animation','Children','Comedy','Crime','Documentary','Drama','Fantasy','Film-Noir','Horror','Musical','Mystery','Romance','Sci-Fi','Thriller','War','Western']
dfItem = pd.read_csv('./ml-100k/ml-100k/u.item', sep='|', names=headerItem)
R = loadRatings()

feaOfUser , feaOfItem = getFea(methodtype)

ShowUser(Fea = feaOfUser['HSR2_1'],sampleNumber=5,cluster=5,method_type='HSR2_1')
ShowUser(Fea = feaOfUser['HSR2_2'],sampleNumber=5,cluster=5,method_type='HSR2_2')

ShowItem(Fea = feaOfItem['HSR2_1'],sampleNumber=5,cluster=5,method_type='HSR2_1')
ShowItem(Fea = feaOfItem['HSR2_2'],sampleNumber=5,cluster=5,method_type='HSR2_2')