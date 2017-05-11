#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2017/4/28 14:33
# @Author  : ConanCui
# @Site    : 
# @File    : UCF.py
# @Software: PyCharm Community Edition
import numpy as np
import pandas as pd
from sklearn import cross_validation as cv
import matplotlib.pyplot as plt

header = ['user_id', 'item_id', 'rating', 'timestamp']
df = pd.read_csv('./ml-100k/ml-100k/u.data', sep='\t', names=header)
n_users = df.user_id.unique().shape[0]
n_items = df.item_id.unique().shape[0]
print 'Number of users = ' + str(n_users) + ' | Number of movies = ' + str(n_items)

train_data, test_data = cv.train_test_split(df, test_size=0.6)
train_data = pd.DataFrame(train_data)
test_data = pd.DataFrame(test_data)

# # Create training and test matrix
# R = np.zeros((n_users, n_items))
# for line in train_data.itertuples():
#     R[line[1] - 1, line[2] - 1] = line[3]
#
# T = np.zeros((n_users, n_items))
# for line in test_data.itertuples():
#     T[line[1] - 1, line[2] - 1] = line[3]

n_users = 8
n_items = 4
R = np.array([[5,0,0,0],[1,0,0,0],[0,0,0,1],[0,0,0,5],[5,0,0,0],[1,0,0,0],[0,0,0,1],[0,0,0,5]])
T = np.array([[5,0,0,0],[1,0,0,0],[0,0,0,1],[0,0,0,5],[5,0,0,0],[1,0,0,0],[0,0,0,1],[0,0,0,5]])

# Index matrix for training data
I = R.copy()
I[I > 0] = 1
I[I == 0] = 0

# Index matrix for test data
I2 = T.copy()
I2[I2 > 0] = 1
I2[I2 == 0] = 0

def prediction(P, Q):
    return np.dot(P, Q)

k  = 5
m,n = R.shape

# Calculate the RMSE
def rmse(I, R, prediction):
    return np.sqrt(np.sum((I * (R - prediction)) ** 2) / len(R[R > 0]))


# 用户相似度矩阵,其中i,j个元素为第i个用户和第j个用户之间的相似度，为对称的矩阵
similarity = np.zeros((n_users, n_users))

den = np.zeros((n_users, 1))
for i in range(n_users):
    # NozerNumber[i] = np.sum([I[i,:]==1])
    # den[i] = np.sqrt(np.sum([R[i,:]**2]))
    den[i] = np.sqrt(np.sum(I[i, :]))

for i in range(n_users):
    for j in range(i+1,n_users):
        mol = np.dot(I[i,:],I[j,:])
        # mol = np.dot(R[i, :], R[j, :])
        if mol == 0:
            continue
        similarity[i, j] = mol / (den[i] * den[j])

similarity = similarity.T + similarity

# 为每一个用户找出K个最相近的用户集合
# simUser = {i : sorted(similarity[i,:])[-1:len(similarity[i,:])-k-1:-1] for i in range(n_users)}
simUser = {i : np.argsort(similarity[i,:])[-1:n_users-k-1:-1] for i in range(n_users)}
# 利用K个最近的用户，为用户i的所有物品产生预测的评分
pre = np.zeros((n_users, n_items))
for i in range(n_users):
    for j in simUser[i]:
        pre[i,:] = pre[i,:] + R[j,:] * similarity[i,j]
        # print ("与第%d用户最相似的%d用户为%d,%d" %(i,k,simUser[i][0],simUser[i][1]))
        # print R[i,:],R[j,:]

pre = (pre - np.min(pre)) / (np.max(pre) - np.min(pre)) * 5
print pre
print rmse(I2, T, pre)
a = 1