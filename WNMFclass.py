import numpy  as np
def WNMF(R,k = 20):
    # Index matrix for training data
    I = R.copy()
    I[I > 0] = 1
    I[I == 0] = 0

    # paramets
    m,n = R.shape
    lamda = 8
    n_epochs = 150 # Number of epochs
    # use stochastic initialization
    P = np.random.rand(m, k) + 10**-4  # Latent user feature matrix
    Q = np.random.rand(k, n) + 10**-4  # Latent movie feature matrix

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





