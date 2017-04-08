#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2017/3/30 10:25
# @Author  : ConanCui
# @Site    : 
# @File    : kmeans.py
# @Software: PyCharm Community Edition

import numpy as np
import pandas as pd
import random
from sklearn.cluster import KMeans
from sklearn import metrics
import time

import HSRx2main
import SGD_WR as SGD
import WNMFclass
import NMFclass
import ALS_WR


def Kmeans(U,n_clusters):
    kmeans = KMeans(n_clusters=n_clusters, precompute_distances=False, random_state=9)
    y_pred = kmeans.fit_predict(U)
    cluster_centers_ = kmeans.cluster_centers_
    return y_pred , cluster_centers_



rnames = ['user_id', 'movie_id', 'rating', 'timestamp']
cf = pd.read_table('./ml-20m/ml-20m/ratings.csv', sep=',', names=rnames,skiprows = 1)

n_users = cf.user_id.unique().shape[0]
n_items = cf.movie_id.unique().shape[0]
# Create training matrix
R = np.zeros((n_users, n_items))

num_movie = 0
num_user = 0
movie = {}
user = {}
i = 0
beginall = time.time()
for line in cf.itertuples():
    begin = time.time()
    # if not (movie_id in movie.keys()):
    if not (movie.has_key(line[2])):
        movie.update({line[2] :num_movie})
        num_movie = num_movie + 1
    if not (user.has_key(line[1])):
        user.update({line[1]:num_user})
        num_user = num_user + 1
    R[user[line[1]], movie[line[2]]] = line[3]
    end = time.time()
    i = i +1
    if i//10000 == 0:
        print ("当前循环时间为 %f 已经循环了百分之： %f" %((end - begin),i/20000263.0*100 ))


endall = time.time()
print endall - beginall
U, V_WNMF = WNMFclass.WNMF(R, k=20, lamda=0)
y_pred , cluster_centers_ = Kmeans(V_WNMF.T,10)

