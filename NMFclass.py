import numpy as np
import activeFunction as af
import matplotlib.pyplot as plt
def NMF(R,k = 20):
    #
    m,n = R.shape
    P = 3 * np.random.rand(m, k)/np.sqrt(m*k) + 10 ** -9  # Latent user feature matrix
    Q = 3 * np.random.rand(k, n)/np.sqrt(k*n) + 10 ** -9   # Latent movie feature matrix
    n_epochs = 150 # Number of epochs
    for epoch in xrange(n_epochs):
        # updata Q
        PTR = np.dot(P.T,R)
        PTPQ = np.dot(np.dot(P.T,P),Q) + 10**-9
        Q = Q* PTR/PTPQ

        # updata P
        RQT = np.dot(R,Q.T)
        PQQT = np.dot(np.dot(P,Q),Q.T) + 10**-9
        P = P * RQT/PQQT
    return P,Q

def rmse(I, X, U, V):
    return np.sqrt(np.sum((I * (X - np.dot(U, V))) ** 2) / len(X[X > 0]))

class nmf(af.activationFunction):
    def __init__(self,type, n_epochs_nmf = 150,gama =1, beta =1  ):
        af.activationFunction.__init__(self, gama, beta, type)
        self.n_epochs_nmf = n_epochs_nmf

    def NMF(self,R,k):
        m, n = R.shape
        P = np.random.rand(m, k)/np.sqrt(m*k) + 10 ** -9  # Latent user feature matrix
        Q = np.random.rand(k, n)/np.sqrt(k*n) + 10 ** -9  # Latent movie feature matrix
        train_errors = []
        for epoch in xrange(self.n_epochs_nmf):
            # updata Q
            PQ = np.dot(P,Q)
            derivate = self.derivative(PQ)
            PTR = np.dot(P.T, R * derivate)
            PTPQ = np.dot(P.T , self.fun(PQ) * derivate) + 10 ** -9
            Q = Q * PTR / PTPQ

            # updata P

            PQ = np.dot(P, Q)
            derivate = self.derivative(PQ)

            RQT = np.dot(R* derivate, Q.T)
            PQQT = np.dot(self.fun(PQ) * derivate, Q.T) + 10 ** -9
            P = P * RQT / PQQT
            train_rmse = np.sqrt(np.sum(((R - self.fun(np.dot(P, Q)))) ** 2) / len(R[R > 0]))
            train_errors.append(train_rmse)
            R_pre = self.fun(np.dot(P,Q))
        # return P, Q , R_pre , train_errors
        return P, Q

if __name__ == '__main__':
    nmf1 = nmf(n_epochs_nmf = 150,gama =0.1, beta =100 , type = 'tanh')
    R = np.array([[1,3,4,5,6,7],[7,6,5,3,2,1]])
    k = 20
    P , Q ,R_pre , train_errors = nmf1.NMF(R,k)
    plt.figure(1)
    plt.plot(range(len(train_errors)), train_errors, marker='o', label='Training Data');
    plt.text(len(train_errors) - 1, train_errors[-1], str(train_errors[-1]), horizontalalignment='center',
             verticalalignment='top')
    plt.title('HSR Learning Curve and K = 20')
    plt.xlabel('Number of Epochs');
    plt.ylabel('RMSE');
    plt.legend()
    plt.grid()
    print R
    print R_pre
    plt.show()






