import numpy  as np
import pandas as pd
from sklearn import cross_validation as cv
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

def prediction(P, Q):
    return np.dot(P, Q)

k  = 20
m,n = R.shape
lamda = 8

# use stochastic initialization
P = 1.5 * np.random.rand(m, k) + 10**-4  # Latent user feature matrix
Q = 1.5 * np.random.rand(k, n) + 10**-4# Latent movie feature matrix
n_epochs = 100 # Number of epochs
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
    return np.sqrt(np.sum((I * (R - FunX2.Fun(prediction(P,Q))) ** 2) / len(R[R > 0])))

class NonlinearFunction:
    def __init__(self):
        pass
    # def Fun(self,X):
    #     return (X)*(X)
    # def Derivative(self,X):
    #     return 2*(X)
    # # def Fun(self,X):
    # #     return (X)
    # # def Derivative(self,X):
    # #     return 1
    def Fun(self,X):
        return 1.5 * (np.exp(X) - np.exp(-X))/(np.exp(X) + np.exp(-X))
    def Derivative(self,X):
        temp = 1.5 * (np.exp(X) - np.exp(-X))/(np.exp(X) + np.exp(-X))
        return 1.5*1.5 - temp**2

FunX2 = NonlinearFunction()
for epoch in xrange(n_epochs):

    # updata P
    PQ = np.dot(P,Q)
    RQT = np.dot(I*R*FunX2.Derivative(PQ),Q.T)
    WPQQT = np.dot(I*FunX2.Fun(PQ)*FunX2.Derivative(PQ),Q.T) + lamda * P + 10**-9
    P = P * (RQT/WPQQT)
   # updata Q
    PQ = np.dot(P, Q)
    PTR = np.dot(P.T,R*I*FunX2.Derivative(PQ))
    PTIPQ = np.dot(P.T,I*FunX2.Fun(PQ)*FunX2.Derivative(PQ))+ lamda * Q + 10**-9
    Q = Q * (PTR/PTIPQ)
    PQ = np.dot(P, Q)

    lossregu = lamda * (np.sum(P ** 2) + np.sum(Q ** 2))
    lossmain = np.sum((I * (R[:, :] - FunX2.Fun(PQ)))**2)
    loss = lossregu+lossmain
    # converged
    ferr[epoch] = loss
    if epoch > 1 :
        derr = np.abs(ferr[epoch] - ferr[epoch - 1]) / m

        if derr < 10**-9:
            break
    train_rmse = rmse(I,R,Q,P)
    test_rmse = rmse(I2,T,Q,P)
    train_errors.append(train_rmse)
    test_errors.append(test_rmse)
    loss_.append(loss)
    loss_regu.append(lossregu)
    loss_main.append(lossmain)
    print epoch, "test_rmse", test_rmse, "train_rmse", train_rmse

plt.figure(1)
plt.plot(range(len(train_errors)), train_errors, marker='o', label='Training Data');
plt.plot(range(len(test_errors)), test_errors, marker='v', label='Test Data');
plt.text(len(train_errors) - 1, train_errors[-1], str(train_errors[-1]),horizontalalignment='center',verticalalignment='top')
plt.text(len(train_errors) - 1, test_errors[-1], str(test_errors[-1]),horizontalalignment='center',verticalalignment='top')

plt.title('WNMF-nl Learning Curve and K = 20')
plt.xlabel('Number of Epochs');
plt.ylabel('RMSE');
plt.legend()
plt.grid()
plt.show()
plt.figure(1)
plt.plot(range(len(loss_regu)), loss_regu, marker='o', label='loss_regu ');
plt.plot(range(len(loss_main)), loss_main, marker='v', label='loss_main');
plt.plot(range(len(loss_)), loss_, marker='v', label='loss');
plt.title('Loss Curve and K = 20')
plt.xlabel('Number of Epochs');
plt.ylabel('Loss');
plt.legend()
plt.grid()
plt.show()


print R
print np.dot(P,Q)

