# -*- coding: utf-8 -*-
import numpy  as np
import pandas as pd
from sklearn import cross_validation as cv
# import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
import copy
import time
import csv

import activeFunction as af
import NMFclass
import WNMFclass

def prediction(U, V):
    return (np.dot(U, V))

def rmse(I, X, U, V):
    return np.sqrt(np.sum((I * (X - prediction(U, V))  ) ** 2) / len(X[X > 0]))

def myrange(begin, end, step):
    if step == 1:
        return range(begin, end + 1, step)
    if step == -1:
        return range(begin, end - 1, step)

class HSR(WNMFclass.wnmf, NMFclass.nmf, af.activationFunction):
    def __init__(self, n_epochs_wnmf, lamda_wnmf, n_epochs_nmf, beta, gama, type):
        af.activationFunction.__init__(self, gama, beta, type)
        NMFclass.nmf.__init__(self, beta=beta, gama=gama, type=type, n_epochs_nmf=n_epochs_nmf)
        WNMFclass.wnmf.__init__(self, n_epochs_wnmf=n_epochs_wnmf,lamda_wnmf=lamda_wnmf)

    def Loaddata(self,usingData = "movielens",test_size=0.4):
        #使用豆瓣数据集
        if usingData == "douban":
            header = ['movie_id', 'movie_name','user_id','user_name', 'rating', 'tag']
            df = pd.read_csv('./Douban/Bigcommentprocess.csv', sep=',', names=header)
            n_users = df.user_id.unique().shape[0]
            n_items = df.movie_id.unique().shape[0]
            print 'Number of users = ' + str(n_users) + ' | Number of movies = ' + str(n_items)
            train_data, test_data = cv.train_test_split(df, test_size=test_size)
            train_data = pd.DataFrame(train_data)
            test_data = pd.DataFrame(test_data)
            # 使用豆瓣数据集
            # Create training and test matrix
            self.X = np.zeros((n_users, n_items))
            for line in train_data.itertuples():
                self.X[line[3] - 1, line[1] - 1] = line[5] / 10
            self.T = np.zeros((n_users, n_items))
            for line in test_data.itertuples():
                self.T[line[3] - 1, line[1] - 1] = line[5] / 10

        if usingData == "movielens":
            header = ['user_id', 'item_id', 'rating', 'timestamp']
            df = pd.read_csv('./ml-100k/ml-100k/u.data', sep='\t', names=header)
            n_users = df.user_id.unique().shape[0]
            n_items = df.item_id.unique().shape[0]
            print 'Number of users = ' + str(n_users) + ' | Number of movies = ' + str(n_items)
            train_data, test_data = cv.train_test_split(df, test_size=test_size)
            train_data = pd.DataFrame(train_data)
            test_data = pd.DataFrame(test_data)
            # Create training and test matrix
            self.X = np.zeros((n_users, n_items))
            for line in train_data.itertuples():
                self.X[line[1] - 1, line[2] - 1] = line[3]

            self.T = np.zeros((n_users, n_items))
            for line in test_data.itertuples():
                self.T[line[1] - 1, line[2] - 1] = line[3]

        # Index matrix for training data
        self.I = self.X.copy()
        self.I[self.I > 0] = 1
        self.I[self.I == 0] = 0

        # Index matrix for test data
        self.I2 = self.T.copy()
        self.I2[self.I2 > 0] = 1
        self.I2[self.I2 == 0] = 0

    def Setparamets(self, M=[20, 100], N=[20, 1000], n_epochs=120, alpha=0.5, lamda=8):
        self.d = M[0]
        m, n = self.X.shape
        self.n_epochs = n_epochs
        self.alpha = alpha
        # each layer parameter
        self.M0 = M
        self.N0 = N

        self.M = [m]
        self.M.extend(M[1:])
        self.M.append(M[0])

        self.N = [n]
        self.N.extend(N[1:])
        self.N.append(N[0])

        self.lamda = lamda
        self.p = len(self.M) - 1
        self.q = len(self.N) - 1
        self.name = "n_epochs " + str(self.n_epochs) + " M " + str(self.M0) +  " N " + str(self.N0) +" " +self.type +  " gama " + str(
                self.gama) +  " beta " +  str(self.beta) +  " lamda " + str(self.lamda) + " lamda_wnmf " + str(self.lamda_wnmf)

    def Initialization(self):
        self.U = {}
        self.V = {}
        self.V_ = {}
        self.U_ = {}
        # initialize U and U_
        for i in myrange(1, self.p, 1):
            self.U[i] = np.random.rand(self.M[i - 1], self.M[i]) / np.sqrt(self.M[i - 1] * self.M[i]) + 10 ** -9
            self.U_[i] = np.random.rand(self.M[i - 1], self.d) / np.sqrt(self.M[i - 1] * self.d) + 10 ** -9

        # initialize U and U_
        for i in myrange(1, self.q, 1):
            self.V[i] = np.random.rand(self.N[i], self.N[i - 1]) / np.sqrt(self.N[i] * self.N[i - 1]) + 10 ** -9
            self.V_[i] = np.random.rand(self.d, self.N[i - 1]) / np.sqrt(self.d * self.N[i - 1]) + 10 ** -9

        # #WNMF
        self.U_[1], self.V_[1] = self.WNMF(self.X, self.d)

        # NMF for U and V
        for i in myrange(1, self.p - 1, 1):
            self.U[i], self.U_[i + 1] = self.NMF(self.U_[i], self.M[i])

        for i in myrange(1, self.q - 1, 1):
            self.V_[i + 1], self.V[i] = self.NMF(self.V_[i], self.N[i])

        self.U[self.p] = copy.deepcopy(self.U_[self.p])
        self.V[self.q] = copy.deepcopy(self.V_[self.q])

        self.mol_U_ = copy.deepcopy(self.U_)
        self.mol_U = copy.deepcopy(self.U)
        self.mol_V = copy.deepcopy(self.V)
        self.mol_V_ = copy.deepcopy(self.V_)

        self.den_U_ = copy.deepcopy(self.U_)
        self.den_U = copy.deepcopy(self.U)
        self.den_V = copy.deepcopy(self.V)
        self.den_V_ = copy.deepcopy(self.V_)

    def Forward_propagation_U(self):
        for i in myrange(self.p, 1, -1):
            if i == len(self.U):
                self.U_[i] = copy.deepcopy(self.U[i])
            else:
                self.U_[i] = self.fun(np.dot(self.U[i], self.U_[i + 1]))

    def Forward_propagation_V(self):
        for i in myrange(self.q, 1, -1):
            if i == len(self.V):
                self.V_[i] = copy.deepcopy(self.V[i])
            else:
                self.V_[i] = self.fun(np.dot(self.V_[i + 1], self.V[i]))

    def Back_Propagation_V(self, j):

        for i in myrange(1, self.q, 1):
            if i == 1:
                self.mol_V_[i] = np.dot(self.U_[i].T, self.X)
                self.den_V_[i] = np.dot(self.U_[i].T, np.dot(self.U_[i], self.V_[i]) * self.I)
            else:
                # derivitavieTemp = self.derivative(V_[i - 1])
                derivitavieTemp = self.derivative(np.dot(self.V_[i], self.V[i - 1]))
                self.mol_V_[i] = np.dot(self.mol_V_[i - 1] * derivitavieTemp, self.V[i - 1].T)
                self.den_V_[i] = np.dot(self.den_V_[i - 1] * derivitavieTemp, self.V[i - 1].T)

            if i == j:
                if i == self.q:
                    self.mol_V[i] = copy.deepcopy(self.mol_V_[i])
                    self.den_V[i] = self.den_V_[i] + 10 ** -9 + self.lamda * self.V[i]
                    break
                else:
                    # derivitavieTemp = self.derivative(V_[i])
                    derivitavieTemp = self.derivative(np.dot(self.V_[i + 1], self.V[i]))
                    self.mol_V[i] = np.dot(self.V_[i + 1].T, self.mol_V_[i] * derivitavieTemp)
                    self.den_V[i] = np.dot(self.V_[i + 1].T, self.den_V_[i] * derivitavieTemp) + 10 ** -9 + self.lamda * \
                                                                                                            self.V[i]
                    break

    def Back_Propagation_U(self, i):
        """
        :param i: 当前更新值

        """
        for j in myrange(1, self.p, 1):
            if j == 1:
                self.mol_U_[j] = np.dot(self.X, self.V_[j].T)
                self.den_U_[j] = np.dot(np.dot(self.U_[j], self.V_[j]) * self.I, self.V_[j].T)
            else:
                derivitavieTemp = self.derivative(np.dot(self.U[j - 1], self.U_[j]))
                # derivitavieTemp = self.derivative(U_[j - 1])
                self.mol_U_[j] = np.dot(self.U[j - 1].T, self.mol_U_[j - 1] * derivitavieTemp)
                self.den_U_[j] = np.dot(self.U[j - 1].T, self.den_U_[j - 1] * derivitavieTemp)

            if j == i:
                if j == self.p:
                    self.mol_U[j] = copy.deepcopy(self.mol_U_[j])
                    self.den_U[j] = self.den_U_[j] + 10 ** -9 + self.lamda * self.U[i]
                    break
                else:
                    # derivitavieTemp = self.derivative(U_[j])
                    derivitavieTemp = self.derivative(np.dot(self.U[j], self.U_[j + 1]))
                    self.mol_U[j] = np.dot(self.mol_U_[j] * derivitavieTemp, self.U_[j + 1].T)
                    self.den_U[j] = np.dot(self.den_U_[j] * derivitavieTemp, self.U_[j + 1].T) + 10 ** -9 + self.lamda * \
                                                                                                            self.U[j]
                    break

    def Factorization(self):
        # factorization
        self.log = []
        self.ferr = np.zeros(self.n_epochs)
        self.terr = np.zeros(self.n_epochs)
        self.train_errors = []
        self.test_errors = []
        self.steprecoderforU1 = []
        self.steprecoderforU2 = []
        self.steprecoderforV1 = []
        self.steprecoderforV2 = []
        for epoch in xrange(self.n_epochs):
            stepU1 = 0
            stepU2 = 0
            stepV1 = 0
            stepV2 = 0
            # updata Vj
            for j in myrange(1, self.q, 1):
                self.Forward_propagation_V()
                self.Back_Propagation_V(j)
                self.V[j] = self.V[j] * ((self.mol_V[j] / self.den_V[j]) ** self.alpha)
                if j== 1:
                    stepV1 = stepV1 + (((self.mol_V[j] / self.den_V[j]) ** self.alpha).mean())
                if j == 2:
                    stepV2 = stepV2 + (((self.mol_V[j] / self.den_V[j]) ** self.alpha).mean())

            # updata Ui
            for i in myrange(self.p, 1, -1):
                self.Forward_propagation_U()
                self.Back_Propagation_U(i)
                self.U[i] = self.U[i] * ((self.mol_U[i] / self.den_U[i]) ** self.alpha)
                if i== 1:
                    stepU1 = stepU1 + (((self.mol_U[i] / self.den_U[i]) ** self.alpha).mean())
                if i == 2:
                    stepU2 = stepU2 + (((self.mol_U[i] / self.den_U[i]) ** self.alpha).mean())


            # convergence
            train_rmse = rmse(self.I, self.X, self.U_[1], self.V_[1])
            test_rmse = rmse(self.I2, self.T, self.U_[1], self.V_[1])
            self.train_errors.append(train_rmse)
            self.test_errors.append(test_rmse)
            self.ferr[epoch] = train_rmse
            self.terr[epoch] = test_rmse

            self.steprecoderforU1.append(stepU1)
            self.steprecoderforU2.append(stepU2)
            self.steprecoderforV2.append(stepV2)
            self.steprecoderforV1.append(stepV1)

            print epoch, "in HSR (",self.name,") test_rmse:", test_rmse, "train_rmse: ", train_rmse
            self.log.append(str(epoch) + self.name + " test_rmse: " + str(test_rmse) + " train_rmse: "+ str(train_rmse))
            if epoch > 1:
                dferr = - self.ferr[epoch] + self.ferr[epoch - 1]
                dterr = - self.terr[epoch] + self.terr[epoch - 1]
                if dferr < np.finfo(float).eps :
                # if dferr < np.finfo(float).eps or dterr < np.finfo(float).eps:
                    break

    def Monitor(self,save=False, show=True, savelog =True):

        plt.figure(1)
        plt.plot(range(len(self.train_errors)), self.train_errors, marker='o', label='Training Data');
        plt.plot(range(len(self.test_errors)), self.test_errors, marker='v', label='Test Data');
        plt.text(len(self.train_errors) - 1, self.train_errors[-1], str(self.train_errors[-1]), horizontalalignment='center',
                 verticalalignment='top')
        plt.text(len(self.train_errors) - 1, self.test_errors[-1], str(self.test_errors[-1]), horizontalalignment='center',
                 verticalalignment='top')

        # plt.title(self.name)
        plt.xlabel('Number of Epochs');
        plt.ylabel('RMSE');
        plt.legend()
        plt.grid()


        plt.figure(2)
        plt.plot(range(len(self.steprecoderforU1)), self.steprecoderforU1, marker='o', label='U_{1}\'s Renewal factor');
        plt.plot(range(len(self.steprecoderforU2)), self.steprecoderforU2, marker='o', label='U_{2}\'s Renewal factor');
        plt.plot(range(len(self.steprecoderforV2)), self.steprecoderforV2, marker='o', label='V_{2}\'s Renewal factor');
        plt.plot(range(len(self.steprecoderforV1)), self.steprecoderforV1, marker='o', label='V_{1}\'s Renewal factor');
        plt.xlabel('Number of Epochs');
        plt.ylabel('The value of Renewal factor');
        plt.legend()


        if save is True:
            figurename = self.name
            plt.savefig(figurename)
            plt.close()
        if show is True:
            plt.show()
        if savelog is True:
            with open("log.csv", 'ab+') as csvfile:
                writer = csv.writer(csvfile,delimiter='\n')
                # for singlecomment in self.log:
                writer.writerow(self.log)
            csvfile.close()
        return self.train_errors[-1], self.test_errors[-1]


class HSRtest(WNMFclass.wnmf, NMFclass.nmf, af.activationFunction):
    def __init__(self, n_epochs_wnmf, lamda_wnmf, n_epochs_nmf, beta, gama, type):
        af.activationFunction.__init__(self, gama, beta, type)
        NMFclass.nmf.__init__(self, beta=beta, gama=gama, type=type, n_epochs_nmf=n_epochs_nmf)
        WNMFclass.wnmf.__init__(self, n_epochs_wnmf=n_epochs_wnmf,lamda_wnmf=lamda_wnmf)

    def Loaddata(self,test_size=0.4):

        header = ['user_id', 'item_id', 'rating', 'timestamp']
        df = pd.read_csv('./ml-100k/ml-100k/u.data', sep='\t', names=header)
        n_users = df.user_id.unique().shape[0]
        n_items = df.item_id.unique().shape[0]
        print 'Number of users = ' + str(n_users) + ' | Number of movies = ' + str(n_items)

        train_data, test_data = cv.train_test_split(df, test_size=0.4)
        train_data = pd.DataFrame(train_data)

        # Create training and test matrix
        self.X = np.zeros((n_users, n_items))
        for line in train_data.itertuples():
            self.X[line[1] - 1, line[2] - 1] = line[3]

        # Index matrix for training data
        self.I = self.X.copy()
        self.I[self.I > 0] = 1
        self.I[self.I == 0] = 0


    def Setparamets(self, M=[20, 100], N=[20, 1000], n_epochs=120, alpha=0.5, lamda=8):
        self.d = M[0]
        m, n = self.X.shape
        self.n_epochs = n_epochs
        self.alpha = alpha
        # each layer parameter
        self.M0 = M
        self.N0 = N

        self.M = [m]
        self.M.extend(M[1:])
        self.M.append(M[0])

        self.N = [n]
        self.N.extend(N[1:])
        self.N.append(N[0])

        self.lamda = lamda
        self.p = len(self.M) - 1
        self.q = len(self.N) - 1
        self.name = "n_epochs " + str(self.n_epochs) + " M " + str(self.M0) +  " N " + str(self.N0) +" " +self.type +  " gama " + str(
                self.gama) +  " beta " +  str(self.beta) +  " lamda " + str(self.lamda) + " lamda_wnmf " + str(self.lamda_wnmf)

    def Initialization(self):
        self.U = {}
        self.V = {}
        self.V_ = {}
        self.U_ = {}
        # initialize U and U_
        for i in myrange(1, self.p, 1):
            self.U[i] = np.random.rand(self.M[i - 1], self.M[i]) / np.sqrt(self.M[i - 1] * self.M[i]) + 10 ** -9
            self.U_[i] = np.random.rand(self.M[i - 1], self.d) / np.sqrt(self.M[i - 1] * self.d) + 10 ** -9

        # initialize U and U_
        for i in myrange(1, self.q, 1):
            self.V[i] = 3 * np.random.rand(self.N[i], self.N[i - 1]) / np.sqrt(self.N[i] * self.N[i - 1]) + 10 ** -9
            self.V_[i] = 3 * np.random.rand(self.d, self.N[i - 1]) / np.sqrt(self.d * self.N[i - 1]) + 10 ** -9

        # #WNMF
        self.U_[1], self.V_[1] = self.WNMF(self.X, self.d)

        # NMF for U and V
        for i in myrange(1, self.p - 1, 1):
            self.U[i], self.U_[i + 1] = self.NMF(self.U_[i], self.M[i])

        for i in myrange(1, self.q - 1, 1):
            self.V_[i + 1], self.V[i] = self.NMF(self.V_[i], self.N[i])

        self.U[self.p] = copy.deepcopy(self.U_[self.p])
        self.V[self.q] = copy.deepcopy(self.V_[self.q])

        self.mol_U_ = copy.deepcopy(self.U_)
        self.mol_U = copy.deepcopy(self.U)
        self.mol_V = copy.deepcopy(self.V)
        self.mol_V_ = copy.deepcopy(self.V_)

        self.den_U_ = copy.deepcopy(self.U_)
        self.den_U = copy.deepcopy(self.U)
        self.den_V = copy.deepcopy(self.V)
        self.den_V_ = copy.deepcopy(self.V_)

    def Forward_propagation_U(self):
        for i in myrange(self.p, 1, -1):
            if i == len(self.U):
                self.U_[i] = copy.deepcopy(self.U[i])
            else:
                self.U_[i] = self.fun(np.dot(self.U[i], self.U_[i + 1]))

    def Forward_propagation_V(self):
        for i in myrange(self.q, 1, -1):
            if i == len(self.V):
                self.V_[i] = copy.deepcopy(self.V[i])
            else:
                self.V_[i] = self.fun(np.dot(self.V_[i + 1], self.V[i]))

    def Back_Propagation_V(self, j):

        for i in myrange(1, self.q, 1):
            if i == 1:
                self.mol_V_[i] = np.dot(self.U_[i].T, self.X)
                self.den_V_[i] = np.dot(self.U_[i].T, np.dot(self.U_[i], self.V_[i]) * self.I)
            else:
                # derivitavieTemp = self.derivative(V_[i - 1])
                derivitavieTemp = self.derivative(np.dot(self.V_[i], self.V[i - 1]))
                self.mol_V_[i] = np.dot(self.mol_V_[i - 1] * derivitavieTemp, self.V[i - 1].T)
                self.den_V_[i] = np.dot(self.den_V_[i - 1] * derivitavieTemp, self.V[i - 1].T)

            if i == j:
                if i == self.q:
                    self.mol_V[i] = copy.deepcopy(self.mol_V_[i])
                    self.den_V[i] = self.den_V_[i] + 10 ** -9 + self.lamda * self.V[i]
                    break
                else:
                    # derivitavieTemp = self.derivative(V_[i])
                    derivitavieTemp = self.derivative(np.dot(self.V_[i + 1], self.V[i]))
                    self.mol_V[i] = np.dot(self.V_[i + 1].T, self.mol_V_[i] * derivitavieTemp)
                    self.den_V[i] = np.dot(self.V_[i + 1].T, self.den_V_[i] * derivitavieTemp) + 10 ** -9 + self.lamda * \
                                                                                                            self.V[i]
                    break

    def Back_Propagation_U(self, i):
        """
        :param i: 当前更新值

        """
        for j in myrange(1, self.p, 1):
            if j == 1:
                self.mol_U_[j] = np.dot(self.X, self.V_[j].T)
                self.den_U_[j] = np.dot(np.dot(self.U_[j], self.V_[j]) * self.I, self.V_[j].T)
            else:
                derivitavieTemp = self.derivative(np.dot(self.U[j - 1], self.U_[j]))
                # derivitavieTemp = self.derivative(U_[j - 1])
                self.mol_U_[j] = np.dot(self.U[j - 1].T, self.mol_U_[j - 1] * derivitavieTemp)
                self.den_U_[j] = np.dot(self.U[j - 1].T, self.den_U_[j - 1] * derivitavieTemp)

            if j == i:
                if j == self.p:
                    self.mol_U[j] = copy.deepcopy(self.mol_U_[j])
                    self.den_U[j] = self.den_U_[j] + 10 ** -9 + self.lamda * self.U[i]
                    break
                else:
                    # derivitavieTemp = self.derivative(U_[j])
                    derivitavieTemp = self.derivative(np.dot(self.U[j], self.U_[j + 1]))
                    self.mol_U[j] = np.dot(self.mol_U_[j] * derivitavieTemp, self.U_[j + 1].T)
                    self.den_U[j] = np.dot(self.den_U_[j] * derivitavieTemp, self.U_[j + 1].T) + 10 ** -9 + self.lamda * \
                                                                                                            self.U[j]
                    break

    def Factorization(self):
        # factorization
        self.log = []
        self.ferr = np.zeros(self.n_epochs)
        self.train_errors = []
        self.steprecoder = []
        for epoch in xrange(self.n_epochs):
            step = 0
            # updata Vj

            for j in myrange(1, self.q, 1):
                self.Forward_propagation_V()
                self.Back_Propagation_V(j)
                self.V[j] = self.V[j] * ((self.mol_V[j] / self.den_V[j]) ** self.alpha)
                step = step + (((self.mol_V[j] / self.den_V[j]) ** self.alpha).mean())
            self.Forward_propagation_V()

            # updata Ui
            for i in myrange(self.p, 1, -1):
                self.Forward_propagation_U()
                self.Back_Propagation_U(i)
                self.U[i] = self.U[i] * ((self.mol_U[i] / self.den_U[i]) ** self.alpha)
                step = step + ((self.mol_U[i] / self.den_U[i]) ** self.alpha).mean()
            self.Forward_propagation_U()
            # convergence
            train_rmse = rmse(self.I, self.X, self.U_[1], self.V_[1])
            self.train_errors.append(train_rmse)
            self.ferr[epoch] = train_rmse
            self.steprecoder.append(step / (self.p * self.q))
            print epoch, "in HSR (",self.name,")" , "train_rmse: ", train_rmse
            if epoch > 1:
                derr = -(self.ferr[epoch] - self.ferr[epoch - 1])
                if derr < np.finfo(float).eps:
                    break

    def Monitor(self,save=False, show=False ,savelog = True):
        plt.figure(1)
        plt.plot(range(len(self.train_errors)), self.train_errors, marker='o', label='Training Data');
        plt.text(len(self.train_errors) - 1, self.train_errors[-1], str(self.train_errors[-1]), horizontalalignment='center',
                 verticalalignment='top')

        plt.title(self.name)
        plt.xlabel('Number of Epochs');
        plt.ylabel('RMSE');
        plt.legend()
        plt.grid()

        if save is True:
            figurename = self.name
            plt.savefig(figurename)
            plt.close()
        if show is True:
            plt.show()


    # plt.figure(2)
    # plt.plot(range(len(self.steprecoder)), self.steprecoder, marker='o', label='Training Data');
    # plt.text(len(self.steprecoder) - 1, self.steprecoder[-1], str(self.steprecoder[-1]), horizontalalignment='center',
    #          verticalalignment='top')
    # plt.title('Step Curve and K = 20')
    # plt.xlabel('Number of Epochs');
    # plt.ylabel('Step');
    # plt.legend()
    # plt.grid()
    # plt.show()

def main(M, N, n_epochs_nmf=150, n_epochs_wnmf=150, lamda_wnmf = 8, gama=1, beta=10, type='linear',n_epochs = 150 , lamda= 8 ,alpha = 0.5 ):
    mf = HSR(n_epochs_nmf =n_epochs_nmf, n_epochs_wnmf =n_epochs_wnmf, lamda_wnmf =lamda_wnmf , gama = gama, beta =beta, type= type)
    mf.Loaddata()
    mf.Setparamets(M=M, N=N,n_epochs = n_epochs , lamda = lamda ,alpha  = alpha)
    start_Real1 = time.time()
    mf.Initialization()
    end_End1 = time.time()
    start_Real2 = time.time()
    mf.Factorization()
    end_End2 = time.time()
    print("initialization: %f real seconds" % (end_End1 - start_Real1))
    print("Factorization: %f real seconds" % (end_End2 - start_Real2))
    train_error, test_error = mf.Monitor()
    return train_error, test_error

def test(M=[20, 100 ,50], N=[20, 1000,500], lamda=8, lamda_wnmf = 8,n_epochs=120, alpha=0.5, gama=1, beta=1, type='linear'):
    mf = HSR(n_epochs_nmf=150, n_epochs_wnmf=150, lamda_wnmf=lamda_wnmf, gama=gama, beta=beta, type=type)
    mf.Loaddata()
    mf.Setparamets(M=M, N=N, lamda=lamda, n_epochs=n_epochs, alpha=alpha)
    start_Real1 = time.time()
    mf.Initialization()
    end_End1 = time.time()
    start_Real2 = time.time()
    mf.Factorization()
    end_End2 = time.time()
    print("initialization: %f real seconds" % (end_End1 - start_Real1))
    print("Factorization: %f real seconds" % (end_End2 - start_Real2))
    train_error, test_error = mf.Monitor()
    return train_error, test_error

if __name__ == '__main__':
    # movielens数据集调整参数
    # 线性最好的 0.933818567767
    main(M=[50,100], N=[50,2000],n_epochs_nmf=150, n_epochs_wnmf=150, lamda_wnmf=8, gama=1, beta=1, type='linear', n_epochs=100, lamda=10)
    # 非线性最好的 0.924
    # main(M=[20, 100 ], N=[20, 1000],n_epochs_nmf=150, n_epochs_wnmf=150, lamda_wnmf=18, gama=1, beta=10, type='tanh', n_epochs=100, lamda=15 ,alpha = 0.5)
    #基础版的效果示意
    # main(M=[50,100], N=[50,2000], n_epochs_nmf=150, n_epochs_wnmf=150, lamda_wnmf=0.01, gama=1, beta=10, type='tanh',
    #     n_epochs=150, lamda=0, alpha=0.5)

    # 豆瓣数据集调整参数
    # main(M=[50, 100], N=[50, 2000], n_epochs_nmf=150, n_epochs_wnmf=150, lamda_wnmf=4, gama=10, beta=10, type='tanh',
    #      n_epochs=100, lamda=20, alpha=0.5)
    # main(M=[50, 100], N=[50, 2000], n_epochs_nmf=150, n_epochs_wnmf=150, lamda_wnmf=4, gama=1, beta=10, type='tanh',
    #      n_epochs=100, lamda=1, alpha=0.5)