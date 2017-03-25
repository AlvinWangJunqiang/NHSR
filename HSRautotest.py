import numpy  as np
import pandas as pd
from sklearn import cross_validation as cv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


import NMFclass
import WNMFclass

def prediction(U, V):
    return np.dot(U, V)

def rmse(I, R, Q, P):
    return np.sqrt(np.sum((I * (R - prediction(P, Q))) ** 2) / len(R[R > 0]))

def myrange(begin,end,step):
    if step == 1:
        return range(begin,end+1,step)
    if step == -1:
        return range(begin,end-1,step)
def HSR(lamda,m1,n1,n_epochs):

    header = ['user_id', 'item_id', 'rating', 'timestamp']
    df = pd.read_csv('./ml-100k/ml-100k/u.data', sep='\t', names=header)
    n_users = df.user_id.unique().shape[0]
    n_items = df.item_id.unique().shape[0]

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
    d  = 20
    # n_epochs = n_epochs
    # lamda = lamda
    m,n = R.shape
    M = (m,m1,d)
    N = (n,n1,d)


    # initialize all
    p = len(M)-1
    q = len(N)-1
    U = {}
    V = {}
    V_ = {}
    U_ = {}

    import time
    start_Real = time.time()

    # initialize U and U_
    for i in myrange(1,p,1):
        U[i] = np.random.rand(M[i-1], M[i]) + 10**-4
        U_[i] = np.random.rand(M[i-1], d) + 10**-4

    # initialize U and U_
    for i in myrange(1,q,1):
        V[i] = np.random.rand(N[i], N[i-1]) + 10**-4
        U_[i] = np.random.rand(d, N[i-1]) + 10**-4

    #WNMF
    U_[1],V_[1] = WNMFclass.WNMF(R,d)

    # NMF for U and V
    for i in myrange(1,p-1,1):
        U[i],U_[i+1] = NMFclass.NMF(U_[i],M[i])

    for i in myrange(1,q-1,1):
        V_[i+1],V[i] = NMFclass.NMF(V_[i],N[i])

    U[p] = U_[p]
    V[q] = V_[q]
    del U_,V_

    # factorization

    ferr = np.zeros(n_epochs)
    train_errors = []
    test_errors = []
    B = {}
    M = {}
    A = {}
    H = {}
    Uall = []
    Vall = []
    for epoch in xrange(n_epochs):
        Uall = np.eye(d,d)
        Vall = np.eye(n,n)

        # updata Vi
        for i in myrange(1,q,1):

            #updata Bi
            B[i] = np.eye(m,m)
            for j in myrange(1,p,1):
                B[i] = np.dot(B[i],U[j])
            if i != q:
                for k in myrange(q,i+1,-1):
                    B[i] = np.dot(B[i],V[k])

            #updata Mi
            M[i] = np.eye(n,n)
            if i != 1:
                for w in myrange(1,i-1,1):
                    M[i] = np.dot(V[w],M[i])

            #updata Vi
            mol = np.dot(np.dot(B[i].T, R * I), M[i].T)
            den = np.dot(np.dot(B[i].T, (I * np.dot(np.dot(B[i], V[i]), M[i]))), M[i].T) + lamda * V[i] + 10 ** -9
            V[i] = V[i] * np.sqrt(mol / den)
            # updata Vall
            Vall = np.dot(V[i],Vall)

        # updata Ui
        for i in myrange(p,1,-1):

            #updata Ai
            A[i] = np.eye(m,m)
            if i != 1:
                for j in myrange(1,i-1,1):
                    A[i] = np.dot(A[i],U[j])
            #updata Hi
            H[i] = np.eye(n,n)
            for k in myrange(1,q,1):
                H[i] = np.dot(V[k],H[i])
            if i != p:
                for w in myrange(p,i+1,-1):
                    H[i] = np.dot(U[w],H[i])
            #updata Ui
            mol = np.dot(np.dot(A[i].T, (I * R)), H[i].T)
            den = np.dot(np.dot(A[i].T, I * np.dot(np.dot(A[i], U[i]), H[i])), H[i].T) + lamda * U[i] + 10 ** -9
            U[i] = U[i] * np.sqrt(mol / den)
            # updata Uall
            Uall = np.dot(U[i],Uall)

        # convergence
        ferr[epoch] = np.sqrt(np.sum((I * (R[:, :] - np.dot(Uall, Vall))) ** 2))
        train_rmse = rmse(I, R, Vall, Uall)
        test_rmse = rmse(I2, T, Vall, Uall)
        train_errors.append(train_rmse)
        test_errors.append(test_rmse)
        print epoch, "in HSR_mn(m1 = ",m1,"n1 = ",n1," lamda = ",lamda,"):test_rmse:", test_rmse, "train_rmse: ",train_rmse
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
    figurename = " m1 "+str(m1) + " n1 "+str(n1)+" lamda "+str(lamda)
    plt.savefig(figurename)
    plt.close()
    return train_errors[-1],test_errors[-1]







