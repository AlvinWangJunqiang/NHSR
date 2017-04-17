import numpy  as np
import HSRx2main

gama = np.array([10])
beta = np.array([10])
lamda = np.array([0.01, 1, 10, 100])
lamda_wnmf = np.array([0.01, 1, 10, 100])
cishu = 10
testrmse = np.eye(cishu, 1)
trainrmse = np.eye(cishu, 1)

for i in range(len(gama)):
    for j in range(len(beta)):
        for k in range(len(lamda)):
            for w in range(len(lamda_wnmf)):
                av_trainrmse = 0
                av_testrmse = 0
                fo = open("result.txt", "a")
                for q in range(cishu):
                    trainrmse[q], testrmse[q] = HSRx2main.test(gama=gama[i], beta=beta[j], type='linear', lamda=lamda[k],
                                                               lamda_wnmf=lamda_wnmf[w])
                    av_testrmse = av_testrmse + testrmse[q]
                    av_trainrmse = av_trainrmse + trainrmse[q]

                av_testrmse = av_testrmse / cishu
                av_trainrmse = av_trainrmse / cishu

                str1 = " gama " + str(gama[i]) + " beta " + str(beta[j]) + " lamda " + str(
                    lamda[k]) + " lamda_wnmf " + str(lamda_wnmf[w]) + " testrmse " + str(
                    testrmse) + " trainrmse " + str(trainrmse) + "\n"
                fo.write(str1);

                str2 = " gama " + str(gama[i]) + " beta " + str(beta[j]) + " lamda " + str(
                    lamda[k]) + " lamda_wnmf " + str(lamda_wnmf[w]) + " testrmse " + str(
                    av_testrmse) + " trainrmse " + str(av_trainrmse) + "\n"
                fo.write(str2);
                fo.close()
