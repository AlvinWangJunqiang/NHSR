#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2017/4/14 21:05
# @Author  : ConanCui
# @Site    : 
# @File    : dataProcess.py
# @Software: PyCharm Community Edition


import pandas as pd
import numpy as np
import os
import csv

def Writecsv(commentcomplete):
    filename = os.getcwd() + "/Douban.csv"
    with open(filename, 'ab+') as csvfile:
        writer = csv.writer(csvfile)
        for singlecomment in commentcomplete:
            writer.writerow(singlecomment)
    csvfile.close()


print ("豆瓣数据集")
header = ['movie_id', 'movie_name', 'user_id', 'user_name', 'rating', 'tag']
df = pd.read_csv('./Douban/Doubanorig.csv', sep=',', names=header)
train_data = pd.DataFrame(df)
n_users = df.user_id.unique().shape[0]
n_items = df.movie_id.unique().shape[0]
print 'Number of users = ' + str(n_users) + ' | Number of movies = ' + str(n_items)
user__ = []
movie__ = []
usernum = {}
movienum = {}
themovie = []
# 用来过滤掉那些同一部电影里用户重复的情况,并且统计用户和电影出现的次数
# 预处理
i = 0
nowmovie = 1
lasmvForuser = {}

for line in train_data.itertuples():
    print i
    # 用户处理
    if not line[3] in user__:
        user__.append(line[3])
        usernum.update({line[3]: 1})
        lasmvForuser.update({line[3]: line[1]})
    else:
        if line[1] != lasmvForuser[line[3]]:
            usernum[line[3]] = usernum[line[3]] + 1
            lasmvForuser[line[3]] = line[1]
    # 电影处理
    if not line[1] in movie__:
        movie__.append(line[1])
    if line[1] == nowmovie:
        themovie.append(line[3])
    else:
        s = pd.Series(themovie)
        movienum.update({nowmovie: s.unique().shape[0]})
        del themovie
        del s
        themovie = [line[3]]
        nowmovie = line[1]
    i = i + 1
s = pd.Series(themovie)
movienum.update({nowmovie: s.unique().shape[0]})



# 过滤评价电影小于20部的用户
user = [i + 1 for i in xrange(n_users) if usernum[user__[i]] >= 30 ]
movie = [j +1 for j in xrange(n_items) if movienum[movie__[j]] >= 30]

# 两个条件同时过滤，并存入csv文件中
user_ = []
movie_ = []
for line in train_data.itertuples():
    componet = []
    if (line[3] in user) and (line[1] in movie):

        if not (line[3] in user_):
            user_.append(line[3])
        if not (line[1] in movie_):
            movie_.append(line[1])

        userid = user_.index(line[3]) + 1
        movieid = movie_.index(line[1]) + 1
        item = [userid,movieid, line[5]/10,line[4],line[2],line[6]]
        componet.append(item)
        Writecsv(componet)
        componet.pop()


n_users = len(user_)
n_items = len(movie_)
print 'Number of users = ' + str(n_users) + ' | Number of movies = ' + str(n_items)
Y = np.zeros((n_users, n_items))
for line in train_data.itertuples():
    if (line[3] in user_) and (line[1] in movie_):
        userid = user_.index(line[3])
        movieid = movie_.index(line[1])
        Y[userid, movieid] = line[5] / 10
num = float(np.nonzero(Y)[0].shape[0])
print("过滤用户后的稀疏度为%f  " %(num/n_users/n_items*100))


