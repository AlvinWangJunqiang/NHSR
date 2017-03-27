import numpy as np

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




