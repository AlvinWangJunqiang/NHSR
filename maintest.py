# -*- coding: UTF-8 -*-
import numpy  as np
import HSRx2main
import time

type = "linear"
gama=[1]
beta=[1]
M = np.array([[50,100],[50,2000]])
N = np.array([[50,100],[50,2000]])
lamda = np.array([0.1,0.5,1,7,10,])
lamda_wnmf = np.array([0.1,0.5,1,7,10])
cishu = 3

begintime = time.time()
Total_number = cishu * len(gama) * len(beta) * len(M) * len(N) * len(lamda) * len(lamda_wnmf)
print ("预计需要 %d 天进行试验" % (Total_number*960//(24*60*60)))
Now_number = 1

testrmse = np.eye(cishu, 1)
trainrmse = np.eye(cishu, 1)

for jj in range(len(beta)):
    for ii in range(len(gama)):
        for i in range(len(M)):
            for j in range(len(N)):
                for k in range(len(lamda)):
                    for w in range(len(lamda_wnmf)):
                        av_trainrmse = 0
                        av_testrmse = 0
                        fo = open("result.txt", "a")
                        for q in range(cishu):
                            trainrmse[q], testrmse[q] = HSRx2main.test(M=M[i], N=N[j], type=type,gama=gama[ii],beta= beta[jj], lamda=lamda[k],
                                                                       lamda_wnmf=lamda_wnmf[w])
                            av_testrmse = av_testrmse + testrmse[q]
                            av_trainrmse = av_trainrmse + trainrmse[q]

                            runtime = time.time() - begintime
                            day = runtime // (60 * 60 * 24)
                            hour = (runtime - day * (60 * 60 * 24)) // (60 * 60)
                            minute = (runtime - day * (60 * 60 * 24) - hour * (60 * 60)) // 60
                            second = (runtime - day * (60 * 60 * 24) - hour * (60 * 60) - minute * 60)
                            print (
                            "****************************************************************************************************************************************")
                            print ("预计需要%d 天 ，已经运行了 %d 天 %d 小时 %d 分钟 %d 秒，运行了%d/%d 个实验" % (Total_number*150//(60 * 60 * 24),
                            day, hour, minute, second, Now_number, Total_number))
                            print (
                            "****************************************************************************************************************************************")

                            Now_number = Now_number + 1
                        av_testrmse = av_testrmse / cishu
                        av_trainrmse = av_trainrmse / cishu

                        # str1 = " M " + str(M[i]) + " N " + str(N[j]) + " lamda " + str(
                        #     lamda[k]) + " lamda_wnmf " + str(lamda_wnmf[w]) + " type "+ type + \
                        #        " gama " + str(gama) + " beta "+ str(beta)+ " testrmse " + str(
                        #     testrmse) + " trainrmse " + str(trainrmse) + "\n"
                        # fo.write(str1);

                        str2 = " M " + str(M[i]) + " N " + str(N[j]) + " lamda " + str(
                            lamda[k]) + " lamda_wnmf " + str(lamda_wnmf[w]) + " type "+ type + \
                               " gama " + str(gama) + " beta "+ str(beta)+ " testrmse " + str(
                            av_testrmse) + " trainrmse " + str(av_trainrmse) + "\n"


                        fo.write(str2);
                        fo.close()
