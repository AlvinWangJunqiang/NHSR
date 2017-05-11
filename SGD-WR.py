# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from sklearn import cross_validation as cv
import matplotlib.pyplot as plt
from sklearn import manifold
from time import time

usingData = "douban"
if usingData == "douban":
    header = ['movie_id', 'movie_name', 'user_id', 'user_name', 'rating', 'tag']
    df = pd.read_csv('./Douban/Bigcommentprocess.csv', sep=',', names=header)
    n_users = df.user_id.unique().shape[0]
    n_items = df.movie_id.unique().shape[0]
    print 'Number of users = ' + str(n_users) + ' | Number of movies = ' + str(n_items)
    train_data, test_data = cv.train_test_split(df, test_size=0.4)
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
    lmbda = 0.01  # Regularisation weight
    k = 50  # Dimension of the latent feature space

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
    lmbda = 0.01
    # lmbda = 0.1  # Regularisation weight
    k = 20  # Dimension of the latent feature space

# Index matrix for training data
I = R.copy()
I[I > 0] = 1
I[I == 0] = 0

# Index matrix for test data
I2 = T.copy()
I2[I2 > 0] = 1
I2[I2 == 0] = 0


# Predict the unknown ratings through the dot product of the latent features for users and items
def prediction(P, Q):
    return np.dot(P.T, Q)



m, n = R.shape  # Number of users and items
n_epochs = 100  # Number of epochs
gamma = 0.01  # Learning rate
np.random.seed(1)
P = 0.001 * np.random.rand(k, m)  # Latent user feature matrix
Q = 0.001 * np.random.rand(k, n)  # Latent movie feature matrix


# Calculate the RMSE
def rmse(I, R, Q, P):
    return np.sqrt(np.sum((I * (R - prediction(P, Q))) ** 2) / len(R[R > 0]))


train_errors = []
test_errors = []

# Only consider non-zero matrix
users, items = R.nonzero()
for epoch in xrange(n_epochs):
    for u, i in zip(users, items):
        try:
            e = R[u, i] - prediction(P[:, u], Q[:, i])  # Calculate error for gradient
            P[:, u] += gamma * (e * Q[:, i] - lmbda * P[:, u])  # Update latent user feature matrix
            Q[:, i] += gamma * (e * P[:, u] - lmbda * Q[:, i])  # Update latent movie feature matrix
        except Warning:
            print(u,i)
    train_rmse = rmse(I, R, Q, P)  # Calculate root mean squared error from train dataset
    test_rmse = rmse(I2, T, Q, P)  # Calculate root mean squared error from test dataset
    train_errors.append(train_rmse)
    test_errors.append(test_rmse)
    print epoch, "test_rmse", test_rmse ,"train_rmse", train_rmse
# Check performance by plotting train and test errors

plt.plot(range(n_epochs), train_errors, marker='o', label='Training Data');
plt.plot(range(n_epochs), test_errors, marker='v', label='Test Data');
plt.text(n_epochs - 1, train_errors[-1], str(train_errors[-1]),horizontalalignment='center',verticalalignment='top')
plt.text(n_epochs - 1, test_errors[-1], str(test_errors[-1]),horizontalalignment='center',verticalalignment='top')
plt.title('SGD-WR Learning Curve and K = 5 ')
plt.xlabel('Number of Epochs');
plt.ylabel('RMSE');
plt.legend()
plt.grid()
plt.show()

# Calculate prediction matrix R_hat (low-rank approximation for R)
R = pd.DataFrame(R)
R_hat = pd.DataFrame(prediction(P, Q))

# Compare true ratings of user 17 with predictions
ratings = pd.DataFrame(data=R.loc[16, R.loc[16, :] > 0]).head(n=5)
ratings['Prediction'] = R_hat.loc[16, R.loc[16, :] > 0]
ratings.columns = ['Actual Rating', 'Predicted Rating']
ratings


X = P.T
n_samples, n_features = X.shape
# Scale and visualize the embedding vectors
def plot_embedding(X, title=None):
    x_min, x_max = np.min(X, 0), np.max(X, 0)
    X = (X - x_min) / (x_max - x_min)

    plt.figure()
    ax = plt.subplot(111)

    for i in range(X.shape[0]):
        plt.text(X[i, 0], X[i, 1], str('o'),
                 color=plt.cm.Set1(4 / 10.),
                 fontdict={'weight': 'bold', 'size': 9})

    shown_flag = np.array([[1., 1.]])  # just something big
    for i in range(n_samples):

        dist = np.sum((X[i] - shown_flag) ** 2, 1)
        if np.min(dist) < 4e-3 :
            # don't show points that are too close
            continue
        plt.text(X[i, 0], X[i, 1], str(1),
                 fontdict={'weight': 'bold', 'size': 9})
        shown_flag = np.r_[shown_flag, [X[i]]]


    plt.xticks([]), plt.yticks([])
    if title is not None:
        plt.title(title)


# -----------------------------------------------------------
# t-SNE embedding of the digits dataset
print("Computing t-SNE embedding")
tsne = manifold.TSNE(n_components=2, init='pca', random_state=0)
t0 = time()
X_tsne = tsne.fit_transform(X)

plot_embedding(X_tsne,
               "t-SNE embedding of the digits (time %.2fs)" %
               (time() - t0))
t1 = time()
print t1 - t0
plt.show()

