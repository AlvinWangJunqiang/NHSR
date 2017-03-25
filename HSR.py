import numpy  as np
import pandas as pd
from sklearn import cross_validation as cv
from sklearn.decomposition import NMF
import matplotlib.pyplot as plt

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

# Index matrix for training data
I = R.copy()
I[I > 0] = 1
I[I == 0] = 0

# Index matrix for test data
I2 = T.copy()
I2[I2 > 0] = 1
I2[I2 == 0] = 0

# paramets
k = 20
m, n = R.shape

lamda_wnmf = 8
n_epochs = 100

p = 2
q = 2

m1 = 100
n1 = 1000


def prediction(U, V):
    return np.dot(U, V)


import time

start_Real = time.time()

# initialize all
U_ = 3 * np.random.rand(m, k) + 10 ** -4  # Latent user feature matrix
V_ = 3 * np.random.rand(k, n) + 10 ** -4  # Latent movie feature matrix

U1 = 3 * np.random.rand(m, m1) + 10 ** -4  # Latent user feature matrix
U2 = 3 * np.random.rand(m1, k) + 10 ** -4  # Latent user feature matrix

V2 = 3 * np.random.rand(k, n1) + 10 ** -4  # Latent user feature matrix
V1 = 3 * np.random.rand(n1, n) + 10 ** -4  # Latent user feature matrix

# WNMF
for epoch in xrange(n_epochs):
    # updata P
    RQT = np.dot(I * R, V_.T)
    WPQQT = np.dot(I * np.dot(U_, V_), V_.T) + lamda_wnmf * U_ + 10 ** -9
    U_ = U_ * (RQT / WPQQT)

    # updata Q
    PTR = np.dot(U_.T, R * I)
    PTIPQ = np.dot(U_.T, I * np.dot(U_, V_)) + lamda_wnmf * V_ + 10 ** -9
    V_ = V_ * (PTR / PTIPQ)

# NMF for U
for epoch in xrange(n_epochs):
    # updata Q
    PTR = np.dot(U1.T, U_)
    PTPQ = np.dot(np.dot(U1.T, U1), U2) + 10 ** -9
    U2 = U2 * PTR / PTPQ

    # updata P
    RQT = np.dot(U_, U2.T)
    PQQT = np.dot(np.dot(U1, U2), U2.T) + 10 ** -9
    U1 = U1 * RQT / PQQT

# NMF for V
for epoch in xrange(n_epochs):
    # updata Q
    PTR = np.dot(V2.T, V_)
    PTPQ = np.dot(np.dot(V2.T, V2), V1) + 10 ** -9
    V1 = V1 * PTR / PTPQ
    # updata P
    RQT = np.dot(V_, V1.T)
    PQQT = np.dot(np.dot(V2, V1), V1.T) + 10 ** -9
    V2 = V2 * RQT / PQQT

ferr = np.zeros(n_epochs)
train_errors = []
test_errors = []


def rmse(I, R, Q, P):
    return np.sqrt(np.sum((I * (R - prediction(P, Q))) ** 2) / len(R[R > 0]))


for epoch in xrange(n_epochs):
    # updata B1 M1
    B1 = np.dot(np.dot(U1, U2), V2)
    M1 = np.eye(n, n)
    # updata V1
    PTR = np.dot(np.dot(B1.T, R * I), M1.T)
    PTIPQ = np.dot(np.dot(B1.T, (I * np.dot(np.dot(B1, V1), M1))), M1.T) + lamda_wnmf * V1 + 10 ** -9
    V1 = V1 * np.sqrt(PTR / PTIPQ)

    # updata B2 M2
    B2 = np.dot(U1, U2)
    M2 = V1
    # updata V2
    PTR = np.dot(np.dot(B2.T, R * I), M2.T)
    PTIPQ = np.dot(np.dot(B2.T, (I * np.dot(np.dot(B2, V2), M2))), M2.T) + lamda_wnmf * V2 + 10 ** -9
    V2 = V2 * np.sqrt(PTR / PTIPQ)

    # updata A2 H2
    A2 = U1
    H2 = np.dot(V2, V1)
    # updata U2
    RQT = np.dot(np.dot(A2.T, (I * R)), H2.T)
    WPQQT = np.dot(np.dot(A2.T, I * np.dot(np.dot(A2, U2), H2)), H2.T) + lamda_wnmf * U2 + 10 ** -9
    U2 = U2 * np.sqrt(RQT / WPQQT)

    # updata A1 H1
    A1 = np.eye(m, m)
    H1 = np.dot(np.dot(U2, V2), V1)
    # updata U1
    RQT = np.dot(np.dot(A1.T, (I * R)), H1.T)
    WPQQT = np.dot(np.dot(A1.T, I * np.dot(np.dot(A1, U1), H1)), H1.T) + lamda_wnmf * U1 + 10 ** -9
    U1 = U1 * np.sqrt(RQT / WPQQT)

    V_ = np.dot(V2, V1)
    U_ = np.dot(U1, U2)
    #
    ferr[epoch] = np.sqrt(np.sum((I * (R[:, :] - np.dot(U_, V_))) ** 2))
    train_rmse = rmse(I, R, V_, U_)
    test_rmse = rmse(I2, T, V_, U_)
    train_errors.append(train_rmse)
    test_errors.append(test_rmse)
    print epoch, "in HSR:test_rmse:", test_rmse ,"train_rmse:", train_rmse
    if epoch > 1:
        derr = np.abs(ferr[epoch] - ferr[epoch - 1]) / n
        if derr < np.finfo(float).eps:
            break
end_End = time.time()
print("Method 1: %f real seconds" % (end_End - start_Real))
plt.plot(range(len(train_errors)), train_errors, marker='o', label='Training Data');
plt.plot(range(len(test_errors)), test_errors, marker='v', label='Test Data');
plt.text(len(train_errors) - 1, train_errors[-1], str(train_errors[-1]), horizontalalignment='center',
         verticalalignment='top')
plt.text(len(train_errors) - 1, test_errors[-1], str(test_errors[-1]), horizontalalignment='center',
         verticalalignment='top')

plt.title('HSR Learning Curve and K = 20')
plt.xlabel('Number of Epochs');
plt.ylabel('RMSE');
plt.legend()
plt.grid()
plt.show()
