import numpy as np
import activeFunction as af
def NMF(R,k = 20):
    #
    m,n = R.shape
    P = 3 * np.random.rand(m, k) + 10**-4  # Latent user feature matrix
    Q = 3 * np.random.rand(k, n) + 10**-4  # Latent movie feature matrix
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

class nmf():
    def __init__(self,n_epochs_nmf = 150,gama =1, beta =1 , type = 'linear' ):
        af.activationFunction.__init__(self, gama, beta, type)
        self.n_epochs_nmf = n_epochs_nmf

    def NMF(self,R,k):
        m, n = R.shape
        P = 3 * np.random.rand(m, k) + 10 ** -4  # Latent user feature matrix
        Q = 3 * np.random.rand(k, n) + 10 ** -4  # Latent movie feature matrix
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
        return P, Q




