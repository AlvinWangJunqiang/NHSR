# -*- coding: utf-8 -*-
import numpy  as np
import pandas as pd
import activeFunction as af
from sklearn import cross_validation as cv
import matplotlib.pyplot as plt
import NMFclass
import WNMFclass
import copy

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


def prediction(U, V):
    return (np.dot(U,V))


def rmse(I, R, U, V):
    return np.sqrt(np.sum((I * (R - prediction(U, V))) ** 2) / len(R[R > 0]))


def myrange(begin, end, step):
    if step == 1:
        return range(begin, end + 1, step)
    if step == -1:
        return range(begin, end - 1, step)


# paramets
d = 20
m, n = R.shape
n_epochs = 150

# each layer parameter
M = (m, 100,d)
N = (n,1000,d)
lamda = 8

# initialize all
p = len(M) - 1
q = len(N) - 1
U = {}
V = {}
V_ = {}
U_ = {}

ferr = np.zeros(n_epochs)
train_errors = []
test_errors = []


class activationFunction:
    def __init__(self):
        self.alpha = 0.5
    # def fun(self, X):
    #     return (X)
    # def derivative(self, X):
    #     return 1
    def fun(self,X):
        return (np.exp(X) - np.exp(-X))/(np.exp(X) + np.exp(-X))
    def derivative(self,X):
        temp = (np.exp(X) - np.exp(-X))/(np.exp(X) + np.exp(-X))
        return 1- temp**2


g = activationFunction()

import time


start_Real1 = time.time()
# initialize U and U_
for i in myrange(1, p, 1):
    U[i] = 3 * np.random.rand(M[i - 1], M[i]) + 10 ** -4
    U_[i] = 3 * np.random.rand(M[i - 1], d) + 10 ** -4

# initialize U and U_
for i in myrange(1, q, 1):
    V[i] = 3 * np.random.rand(N[i], N[i - 1]) + 10 ** -4
    U_[i] = 3 * np.random.rand(d, N[i - 1]) + 10 ** -4

# #WNMF
U_[1], V_[1] = WNMFclass.WNMF(R, d)

# NMF for U and V
for i in myrange(1, p - 1, 1):
    U[i], U_[i + 1] = NMFclass.NMF(U_[i], M[i])

for i in myrange(1, q - 1, 1):
    V_[i + 1], V[i] = NMFclass.NMF(V_[i], N[i])

U[p] = copy.deepcopy(U_[p])
V[q] = copy.deepcopy(V_[q])

mol_U_ = copy.deepcopy(U_)
mol_U = copy.deepcopy(U)
mol_V = copy.deepcopy(V)
mol_V_ = copy.deepcopy(V_)

den_U_ = copy.deepcopy(U_)
den_U = copy.deepcopy(U)
den_V = copy.deepcopy(V)
den_V_ = copy.deepcopy(V_)

end_End1 = time.time()
def forward_propagation_U(U, U_):
    for i in myrange(len(U), 1, -1):
        if i == len(U):
            U_[i] = copy.deepcopy(U[i])
        else:
            U_[i] = g.fun(np.dot(U[i], U_[i + 1]))


def forward_propagation_V(V, V_):
    for i in myrange(len(V), 1, -1):
        if i == len(V):
            V_[i] = copy.deepcopy(V[i])
        else:
            V_[i] = g.fun(np.dot(V_[i + 1], V[i]))


def Back_Propagation_V(j ,mol_V_, mol_V, den_V_, den_V, V, U_, V_, R, I):
    """

    :param j: 当前更新值
    :param mol_V_:
    :param mol_V:
    :param den_V_:
    :param den_V:
    :param V:
    :param U_:
    :param V_:
    :param X:
    :param I:
    :return:
    """
    for i in myrange(1, q, 1):
        if i == 1:
            mol_V_[i] = np.dot(U_[i].T, R)
            den_V_[i] = np.dot(U_[i].T, np.dot(U_[i], V_[i]) * I)
        else:
            # derivitavieTemp = g.derivative(V_[i - 1])
            derivitavieTemp = g.derivative(np.dot(V_[i],V[i-1]))
            mol_V_[i] = np.dot(mol_V_[i - 1] * derivitavieTemp, V[i - 1].T)
            den_V_[i] = np.dot(den_V_[i-1] * derivitavieTemp , V[i - 1].T)

        if i == j:
            if i == q:
                mol_V[i] = copy.deepcopy(mol_V_[i])
                den_V[i] = den_V_[i] + 10**-9 + lamda*V[i]
                break
            else:
                # derivitavieTemp = g.derivative(V_[i])
                derivitavieTemp = g.derivative(np.dot(V_[i+1],V[i]))
                mol_V[i] = np.dot(V_[i+1].T , mol_V_[i] * derivitavieTemp )
                den_V[i] = np.dot(V_[i+1].T, den_V_[i] * derivitavieTemp) + 10**-9 + lamda*V[i]
                break

def Back_Propagation_U( i  ,mol_U_, mol_U, den_U_, den_U, U, U_, V_, R, I):
    """
    :param i: 当前更新值
    :param mol_U_:
    :param mol_U:
    :param den_U_:
    :param den_U:
    :param U:
    :param U_:
    :param V_:
    :param R:
    :param I:
    :return:
    """
    for j in myrange(1, p, 1):
        if j == 1:
            mol_U_[j] = np.dot(R,V_[j].T)
            den_U_[j] = np.dot(np.dot(U_[j],V_[j])*I , V_[j].T)
        else:
            derivitavieTemp = g.derivative(np.dot(U[j-1],U_[j]))
            # derivitavieTemp = g.derivative(U_[j - 1])
            mol_U_[j] = np.dot(U[j - 1].T, mol_U_[j - 1] * derivitavieTemp)
            den_U_[j] = np.dot(U[j - 1].T ,den_U_[j-1] * derivitavieTemp )

        if j == i:
            if j == p:
                mol_U[j] = copy.deepcopy(mol_U_[j])
                den_U[j] = den_U_[j] + 10**-9 + lamda*U[i]
                break
            else:
                # derivitavieTemp = g.derivative(U_[j])
                derivitavieTemp = g.derivative(np.dot(U[j],U[j+1]))
                mol_U[j] = np.dot(mol_U_[j] * derivitavieTemp , U_[j+1].T )
                den_U[j] = np.dot(den_U_[j] * derivitavieTemp , U_[j+1].T ) + 10**-9 + lamda*U[j]
                break

# factorization

steprecoder = []
start_Real2 = time.time()
for epoch in xrange(n_epochs):
    step = 0
    # updata Vj
    for j in myrange(1, q, 1):
        forward_propagation_V(V, V_)
        Back_Propagation_V(j, mol_V_, mol_V, den_V_, den_V, V, U_, V_, R, I)
        V[j] = V[j] * ((mol_V[j]/den_V[j])**0.5)
        step = step + (((mol_V[j]/den_V[j])**0.5).mean())

    # updata Ui
    for i in myrange(p, 1, -1):
        forward_propagation_U(U, U_)
        Back_Propagation_U(i, mol_U_, mol_U, den_U_, den_U, U, U_, V_, R, I)
        U[i] = U[i] * ((mol_U[i] / den_U[i]) ** 0.5)
        step = step + ((mol_U[i] / den_U[i]) ** 0.5).mean()


    # convergence
    train_rmse = rmse(I, R, U_[1], V_[1])
    test_rmse = rmse(I2, T, U_[1], V_[1])
    train_errors.append(train_rmse)
    test_errors.append(test_rmse)
    ferr[epoch] = train_rmse
    steprecoder.append(step/(p*q))
    print epoch, "in HSR:test_rmse:", test_rmse, "train_rmse: ", train_rmse
    if epoch > 1:
        derr = np.abs(ferr[epoch] - ferr[epoch - 1])
        if derr < np.finfo(float).eps:
            break
end_End2 = time.time()

print("initialization: %f real seconds" % (end_End1 - start_Real1))
print("Factorization: %f real seconds" % (end_End2 - start_Real2))


plt.figure(1)
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


plt.figure(2)
plt.plot(range(len(steprecoder)), steprecoder, marker='o', label='Training Data');
plt.text(len(steprecoder) - 1, steprecoder[-1], str(steprecoder[-1]), horizontalalignment='center',
         verticalalignment='top')
plt.title('Step Curve and K = 20')
plt.xlabel('Number of Epochs');
plt.ylabel('Step');
plt.legend()
plt.grid()
plt.show()