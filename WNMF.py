# -*- coding: utf-8 -*-
import numpy  as np
import pandas as pd
from sklearn import cross_validation as cv
import matplotlib.pyplot as plt

usingData = "douban"
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
    k = 20
    m, n = R.shape
    # lamda = 4
    lamda = 0

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

    k = 50
    m, n = R.shape
    # lamda = 8
    lamda = 0

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



# use stochastic initialization
P = 3 * np.random.rand(m, k) + 10**-4  # Latent user feature matrix
Q = 3 * np.random.rand(k, n) + 10**-4# Latent movie feature matrix
n_epochs = 150 # Number of epochs
ferr = np.zeros(n_epochs)
train_errors = []
test_errors = []
loss_ = []
loss_main = []
loss_regu = []
x1 = []
x2 = []
# Calculate the RMSE
def rmse(I, R, Q, P):
    return np.sqrt(np.sum((I * (R - prediction(P, Q))) ** 2) / len(R[R > 0]))


for epoch in xrange(n_epochs):

    # updata P
    RQT = np.dot(I*R,Q.T)
    WPQQT = np.dot(I*np.dot(P,Q),Q.T) + lamda * P + 10**-9
    P = P * (RQT/WPQQT)
    # P /= np.sqrt(np.sum(P ** 2.0, axis=0))
   # updata Q
    PTR = np.dot(P.T,R*I)
    PTIPQ = np.dot(P.T,I*np.dot(P,Q))+ lamda * Q + 10**-9
    Q = Q* (PTR/PTIPQ)
    # Q /= np.sqrt(np.sum(Q ** 2.0, axis=0)) + 10**-9
    lossregu = lamda * (np.sum(P ** 2) + np.sum(Q ** 2))
    lossmain = np.sum((I * (R[:, :] - np.dot(P, Q)))**2)
    loss = lossregu+lossmain
    # converged
    ferr[epoch] = loss
    if epoch > 1 :
        derr = np.abs(ferr[epoch] - ferr[epoch - 1]) / m

        if derr < 10**-9:
            break
    train_rmse = rmse(I,R,Q,P)
    test_rmse = rmse(I2,T,Q,P)
    XX = np.dot(P,Q)
    x1.append(XX.max())
    x2.append(XX.min())
    train_errors.append(train_rmse)
    test_errors.append(test_rmse)
    loss_.append(loss)
    loss_regu.append(lossregu)
    loss_main.append(lossmain)
    print epoch, "test_rmse", test_rmse,"train_rmse", train_rmse,"lossmain", lossmain, "lossregu", lossregu

plt.figure(1)
plt.plot(range(len(train_errors)), train_errors, marker='o', label='Training Data');
plt.plot(range(len(test_errors)), test_errors, marker='v', label='Test Data');
plt.text(len(train_errors) - 1, train_errors[-1], str(train_errors[-1]),horizontalalignment='center',verticalalignment='top')
plt.text(len(train_errors) - 1, test_errors[-1], str(test_errors[-1]),horizontalalignment='center',verticalalignment='top')

plt.title('WNMF Learning Curve and K = 20')
plt.xlabel('Number of Epochs');
plt.ylabel('RMSE');
plt.legend()
plt.grid()
plt.show()

# plt.figure(1)
# plt.plot(range(len(loss_regu)), loss_regu, marker='o', label='loss_regu ');
# plt.plot(range(len(loss_main)), loss_main, marker='v', label='loss_main');
# plt.plot(range(len(loss_)), loss_, marker='v', label='loss');
# plt.title('Loss Curve and K = 20')
# plt.xlabel('Number of Epochs');
# plt.ylabel('Loss');
# plt.legend()
# plt.grid()
# plt.show()

print R
print np.dot(P,Q)

