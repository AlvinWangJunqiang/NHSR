import numpy  as np
import activeFunction as af
def WNMF(R,k = 20,lamda = 8):
    # Index matrix for training data
    I = R.copy()
    I[I > 0] = 1
    I[I == 0] = 0

    # paramets
    m,n = R.shape
    n_epochs = 150 # Number of epochs
    # use stochastic initialization
    P = np.random.rand(m, k) / np.sqrt(m * k) + 10 ** -9  # Latent user feature matrix
    Q = np.random.rand(k, n) / np.sqrt(n * k) + 10 ** -9  # Latent movie feature matrix

    # factorazation
    for epoch in xrange(n_epochs):
        # updata Q
        PTR = np.dot(P.T,R*I)
        PTIPQ = np.dot(P.T,I*np.dot(P,Q))+ lamda * Q + 10**-9
        Q = Q* (PTR/PTIPQ)

        # updata P
        RQT = np.dot(I*R,Q.T)
        WPQQT = np.dot(I*np.dot(P,Q),Q.T) + lamda * P + 10**-9
        P = P * (RQT/WPQQT)
    return P,Q

class wnmf(af.activationFunction):
    def __init__(self,n_epochs_wnmf = 150,lamda_wnmf = 8 ):
        self.n_epochs_wnmf = n_epochs_wnmf
        self.lamda_wnmf = lamda_wnmf
    def WNMF(self,R , k):
        # Index matrix for training data
        I = R.copy()
        I[I > 0] = 1
        I[I == 0] = 0

        # paramets
        m, n = R.shape

        # use stochastic initialization
        P = np.random.rand(m, k)/np.sqrt(m*k) + 10 ** -9  # Latent user feature matrix
        Q = np.random.rand(k, n)/np.sqrt(n*k) + 10 ** -9  # Latent movie feature matrix

        # factorazation
        for epoch in xrange(self.n_epochs_wnmf):
            # updata Q
            PTR = np.dot(P.T, R * I)
            PTIPQ = np.dot(P.T, I * np.dot(P, Q)) + self.lamda_wnmf * Q + 10 ** -9
            Q = Q * (PTR / PTIPQ)

            # updata P
            RQT = np.dot(I * R, Q.T)
            WPQQT = np.dot(I * np.dot(P, Q), Q.T) + self.lamda_wnmf * P + 10 ** -9
            P = P * (RQT / WPQQT)
        return P, Q



if __name__ == '__main__':
    R = np.array([[4,0,4,0],[0,1,0,3],[5,0,0,1],[0,1,2,0]])
    P , Q = WNMF(R,k = 2,lamda=0)
    print P , Q
    print np.dot(P,Q)