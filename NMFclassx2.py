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

def NMF(R,k = 20):

    m,n = R.shape
    P = 3 * np.random.rand(m, k) + 10 ** -4  # Latent user feature matrix
    Q = 3 * np.random.rand(k, n) + 10 ** -4  # Latent movie feature matrix
    n_epochs = 100  # Number of epochs
    FunX2 = NonlinearFunction()

    # factorazation
    for epoch in xrange(n_epochs):

       # updata Q
        PQ = np.dot(P, Q)
        PTR = np.dot(P.T,R*FunX2.Derivative(PQ))
        PTIPQ = np.dot(P.T,FunX2.Fun(PQ)*FunX2.Derivative(PQ))+ 10**-9
        Q = Q * (PTR/PTIPQ) ** 0.8

       # updata P
        PQ = np.dot(P, Q)
        RQT = np.dot(R * FunX2.Derivative(PQ), Q.T)
        WPQQT = np.dot(FunX2.Fun(PQ) * FunX2.Derivative(PQ), Q.T) + 10 ** -9
        P = P * (RQT / WPQQT) ** 0.8


    return P,Q


