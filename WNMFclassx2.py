import numpy  as np


class NonlinearFunction:
    def __init__(self):
        pass

    def Fun(self, X):
        return (X) * (X)

    def Derivative(self, X):
        return 2 * (X)
        # def Fun(self,X):
        #     return (X)
        # def Derivative(self,X):
        #     return 1

def WNMF(R,k = 20):
    # Index matrix for training data
    I = R.copy()
    I[I > 0] = 1
    I[I == 0] = 0

    # paramets
    m,n = R.shape
    lamda = 8
    n_epochs = 100  # Number of epochs
    ferr = np.zeros(n_epochs)
    FunX2 = NonlinearFunction()

    # use stochastic initialization
    P = np.random.rand(m, k) + 10**-4  # Latent user feature matrix
    Q = np.random.rand(k, n) + 10**-4# Latent movie feature matrix

    # factorazation
    for epoch in xrange(n_epochs):
        # updata Q
        PQ = np.dot(P, Q)
        PTR = np.dot(P.T, R * I * FunX2.Derivative(PQ))
        PTIPQ = np.dot(P.T, I * FunX2.Fun(PQ) * FunX2.Derivative(PQ)) + lamda * Q + 10 ** -9
        Q = Q * (PTR / PTIPQ) ** 0.8

        # updata P
        PQ = np.dot(P,Q)
        RQT = np.dot(I*R*FunX2.Derivative(PQ),Q.T)
        WPQQT = np.dot(I*FunX2.Fun(PQ)*FunX2.Derivative(PQ),Q.T) + lamda * P + 10**-9
        P = P * (RQT/WPQQT) ** 0.8
        # converged
        PQ = np.dot(P, Q)
        ferr[epoch] = np.sqrt(np.sum((I * (R[:, :] - FunX2.Derivative(PQ))) ** 2))
        if epoch > 1 :
            derr = np.abs(ferr[epoch] - ferr[epoch - 1]) / m
            if derr < 10**-9:
                break
    return P,Q


