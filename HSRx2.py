import numpy  as np
import pandas as pd
import NMFclassx2
import WNMFclass
import NMFclass
from sklearn import cross_validation as cv
import matplotlib.pyplot as plt

header = ['user_id', 'item_id', 'rating', 'timestamp']
df = pd.read_csv('./ml-100k/ml-100k/u.data', sep='\t', names=header)
n_users = df.user_id.unique().shape[0]
n_items = df.item_id.unique().shape[0]
print 'Number of users = ' + str(n_users) + ' | Number of movies = ' + str(n_items)

train_data, test_data = cv.train_test_split(df, test_size=0.4 )
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
n_epochs = 150

p = 2
q = 2

m1 = 100
n1 = 1000
step = 0.5

def prediction(U, V):
    return np.dot(U, V)


import time



# initialize all
U_ = 3 * np.random.rand(m, k) + 10 ** -4  # Latent user feature matrix
V_ = 3 * np.random.rand(k, n) + 10 ** -4  # Latent movie feature matrix

U1 = 3 * np.random.rand(m, m1) + 10 ** -4  # Latent user feature matrix
U2 = 3 * np.random.rand(m1, k) + 10 ** -4  # Latent user feature matrix

V2 = 3 * np.random.rand(k, n1) + 10 ** -4  # Latent user feature matrix
V1 = 3 * np.random.rand(n1, n) + 10 ** -4  # Latent user feature matrix

# WNMF
U_,V_ = WNMFclass.WNMF(R,k)
# for epoch in xrange(n_epochs):
#     # updata P
#     RQT = np.dot(I * R, V_.T)
#     WPQQT = np.dot(I * np.dot(U_, V_), V_.T) + lamda_wnmf * U_ + 10 ** -9
#     U_ = U_ * (RQT / WPQQT)
#
#     # updata Q
#     PTR = np.dot(U_.T, R * I)
#     PTIPQ = np.dot(U_.T, I * np.dot(U_, V_)) + lamda_wnmf * V_ + 10 ** -9
#     V_ = V_ * (PTR / PTIPQ)

# NMF for U
U1,U2 = NMFclass.NMF(U_,m1)
# for epoch in xrange(n_epochs):
#     # updata Q
#     PTR = np.dot(U1.T, U_)
#     PTPQ = np.dot(np.dot(U1.T, U1), U2) + 10 ** -9
#     U2 = U2 * PTR / PTPQ
#
#     # updata P
#     RQT = np.dot(U_, U2.T)
#     PQQT = np.dot(np.dot(U1, U2), U2.T) + 10 ** -9
#     U1 = U1 * RQT / PQQT

# NMF for V
V2,V1 = NMFclass.NMF(V_,n1)
# for epoch in xrange(n_epochs):
#     # updata Q
#     PTR = np.dot(V2.T, V_)
#     PTPQ = np.dot(np.dot(V2.T, V2), V1) + 10 ** -9
#     V1 = V1 * PTR / PTPQ
#     # updata P
#     RQT = np.dot(V_, V1.T)
#     PQQT = np.dot(np.dot(V2, V1), V1.T) + 10 ** -9
#     V2 = V2 * RQT / PQQT

ferr = np.zeros(n_epochs)
train_errors = []
test_errors = []


def rmse(I, R, Q, P):
    return np.sqrt(np.sum((I * (R - prediction(P, Q))) ** 2) / len(R[R > 0]))

class NonlinearFunction:
    def __init__(self):
        pass
    # #sigmoid
    # def Fun(self,X):
    #     return 1/(1+np.exp(-X))
    # def Derivative(self,X):
    #     temp = 1/(1+np.exp(-X))
    #     return temp*(1-temp)
    ##x2
    # def Fun(self,X):
    #     return (X)*(X)
    # def Derivative(self,X):
    #     return 2*(X)
    # #linear
    def Fun(self,X):
        return (X)
    def Derivative(self,X):
        return 1
    #linear2
    # def Fun(self,X):
    #     return (X*1.5+0.1)
    # def Derivative(self,X):
    #     return 1.5
    # tanh
    # def Fun(self,X):
    #     return 1.7 * (np.exp(X) - np.exp(-X))/(np.exp(X) + np.exp(-X))
    # def Derivative(self,X):
    #     temp = 1.7 * (np.exp(X) - np.exp(-X))/(np.exp(X) + np.exp(-X))
    #     return 1.7*1.7 - temp**2

steprecoder = []
FunX2 = NonlinearFunction()
start_Real = time.time()
for epoch in xrange(n_epochs):
    step_temp = 0

    # step 1 updata V1
    # updata U_ V_
    U_ = FunX2.Fun(np.dot(U1,U2))
    V_ =  FunX2.Fun(np.dot(V2,V1))
    U_V_W = np.dot(U_,V_)*I
    detgV2V1 = FunX2.Derivative(np.dot(V2, V1))
    mol_1 = (np.dot(U_.T,R) * detgV2V1)
    den_1 = (np.dot(U_.T,U_V_W) * detgV2V1)
    # updata V1
    mol = np.dot(V2.T,mol_1)
    den = np.dot(V2.T,den_1) + lamda_wnmf * V1 + 10**-9
    V1 = V1 * (mol / den) ** step
    step_temp = step_temp + (mol / den).mean()


    # step 2 updata V2
    # updata U_ V_
    U_ = FunX2.Fun(np.dot(U1,U2))
    V_ =  FunX2.Fun(np.dot(V2,V1))
    U_V_W = np.dot(U_,V_)*I
    detgV2V1 = FunX2.Derivative(np.dot(V2, V1))
    mol_1 = (np.dot(U_.T,R) * detgV2V1)
    den_1 = (np.dot(U_.T,U_V_W) * detgV2V1)
    # updata V2
    mol = np.dot(mol_1,V1.T)
    den = np.dot(den_1,V1.T) + lamda_wnmf * V2 + 10**-9
    V2 = V2 * (mol / den) ** step
    step_temp = step_temp + (mol / den).mean()


    # step 3 updata U2
    # updata U_ V_
    U_ = FunX2.Fun(np.dot(U1,U2))
    V_ =  FunX2.Fun(np.dot(V2,V1))
    U_V_W = np.dot(U_, V_) * I
    detgU1U2 = FunX2.Derivative(np.dot(U1, U2))
    mol_1 = (np.dot(R,V_.T) * detgU1U2)
    den_1 = (np.dot(U_V_W,V_.T) * detgU1U2)
    # updata U2
    mol = np.dot(U1.T,mol_1)
    den = np.dot(U1.T,den_1) + lamda_wnmf * U2 + 10**-9
    U2 = U2 * (mol / den) ** step
    step_temp = step_temp + (mol / den).mean()

    # step 4 updata U1
    # updata U_ V_
    U_ = FunX2.Fun(np.dot(U1,U2))
    V_ =  FunX2.Fun(np.dot(V2,V1))
    U_V_W = np.dot(U_, V_) * I
    detgU1U2 = FunX2.Derivative(np.dot(U1, U2))
    mol_1 = (np.dot(R,V_.T) * detgU1U2)
    den_1 = (np.dot(U_V_W,V_.T) * detgU1U2)
    # updata U1
    mol = np.dot(mol_1,U2.T)
    den = np.dot(den_1,U2.T) + lamda_wnmf * U1 + 10**-9
    U1 = U1 * (mol / den) ** step
    step_temp = step_temp + (mol / den).mean()

    # Determine if convergence
    U_ = FunX2.Fun(np.dot(U1,U2))
    V_ =  FunX2.Fun(np.dot(V2,V1))
    ferr[epoch] = np.sqrt(np.sum((I * (R[:, :] - np.dot(U_, V_))) ** 2))
    train_rmse = rmse(I, R, V_, U_)
    test_rmse = rmse(I2, T, V_, U_)
    train_errors.append(train_rmse)
    test_errors.append(test_rmse)
    steprecoder.append(step_temp/4)
    print epoch, "in HSR:test_rmse:", test_rmse,"train_rmse:", train_rmse
    if epoch > 1:
        derr = np.abs(ferr[epoch] - ferr[epoch - 1]) / n
        if derr < np.finfo(float).eps:
            break
end_End = time.time()
print("Method 1: %f real seconds" % (end_End - start_Real))

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