# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from sklearn import cross_validation as cv

usingData = "movielens"
if usingData == "douban":
    header = ['movie_id', 'movie_name', 'user_id', 'user_name', 'rating', 'tag']
    df = pd.read_csv('./Douban/Bigcommentprocess.csv', sep=',', names=header)
    n_users = df.user_id.unique().shape[0]
    n_items = df.movie_id.unique().shape[0]
    print 'Number of users = ' + str(n_users) + ' | Number of movies = ' + str(n_items)
    train_data, test_data = cv.train_test_split(df, test_size=0.6)
    train_data = pd.DataFrame(train_data)
    test_data = pd.DataFrame(test_data)
    # 使用豆瓣数据集
    # Create training and test matrix
    R = np.zeros((n_users, n_items))
    for line in train_data.itertuples():
        R[line[3] - 1, line[1] - 1] = line[5] / 10
    T = np.zeros((n_users, n_items))
    for line in test_data.itertuples():
        T[line[3] - 1, line[1] - 1] = line[5] / 10
    # lmbda = 0.1  # Regularisation weight
    lmbda = 0.01  # Regularisation weight
    k = 50  # Dimensionality of latent feature space

if usingData == "movielens":
    header = ['user_id', 'item_id', 'rating', 'timestamp']
    df = pd.read_csv('./ml-100k/ml-100k/u.data', sep='\t', names=header)
    n_users = df.user_id.unique().shape[0]
    n_items = df.item_id.unique().shape[0]
    print 'Number of users = ' + str(n_users) + ' | Number of movies = ' + str(n_items)

    train_data, test_data = cv.train_test_split(df, test_size=0.4)
    train_data = pd.DataFrame(train_data)
    test_data = pd.DataFrame(test_data)

    # Create training and test matrix
    R = np.zeros((n_users, n_items))
    for line in train_data.itertuples():
        R[line[1] - 1, line[2] - 1] = line[3]

    T = np.zeros((n_users, n_items))
    for line in test_data.itertuples():
        T[line[1] - 1, line[2] - 1] = line[3]
    lmbda = 0.00001  # Regularisation weight
    # lmbda = 0.1  # Regularisation weight
    k = 20  # Dimensionality of latent feature space


# Index matrix for training data
I = R.copy()
I[I > 0] = 1
I[I == 0] = 0

# Index matrix for test data
I2 = T.copy()
I2[I2 > 0] = 1
I2[I2 == 0] = 0

# Calculate the RMSE
def rmse(I,R,Q,P):
    return np.sqrt(np.sum((I * (R - np.dot(P.T,Q)))**2)/len(R[R > 0]))


m, n = R.shape # Number of users and items
n_epochs = 15 # Number of epochs

P = 3 * np.random.rand(k,m) # Latent user feature matrix
Q = 3 * np.random.rand(k,n) # Latent movie feature matrix
Q[0,:] = R[R != 0].mean(axis=0) # Avg. rating for each movie
E = np.eye(k) # (k x k)-dimensional idendity matrix

train_errors = []
test_errors = []

# Repeat until convergence
for epoch in range(n_epochs):
    # Fix Q and estimate P
    for i, Ii in enumerate(I):
        nui = np.count_nonzero(Ii)  # Number of items user i has rated
        if (nui == 0): nui = 1  # Be aware of zero counts!

        # Least squares solution
        Ai = np.dot(Q, np.dot(np.diag(Ii), Q.T)) + lmbda * nui * E
        Vi = np.dot(Q, np.dot(np.diag(Ii), R[i].T))
        P[:, i] = np.linalg.solve(Ai, Vi)

    # Fix P and estimate Q
    for j, Ij in enumerate(I.T):
        nmj = np.count_nonzero(Ij)  # Number of users that rated item j
        if (nmj == 0): nmj = 1  # Be aware of zero counts!

        # Least squares solution
        Aj = np.dot(P, np.dot(np.diag(Ij), P.T)) + lmbda * nmj * E
        Vj = np.dot(P, np.dot(np.diag(Ij), R[:, j]))
        Q[:, j] = np.linalg.solve(Aj, Vj)

    train_rmse = rmse(I, R, Q, P)
    test_rmse = rmse(I2, T, Q, P)
    train_errors.append(train_rmse)
    test_errors.append(test_rmse)

    print "[Epoch %d/%d] train error: %f, test error: %f" \
          % (epoch + 1, n_epochs, train_rmse, test_rmse)

print "Algorithm converged"

# Check performance by plotting train and test errors
import matplotlib.pyplot as plt

plt.plot(range(n_epochs), train_errors, marker='o', label='Training Data');
plt.plot(range(n_epochs), test_errors, marker='v', label='Test Data');
plt.title('ALS-WR Learning Curve')
plt.xlabel('Number of Epochs');
plt.ylabel('RMSE');
plt.legend()
plt.grid()
plt.show()