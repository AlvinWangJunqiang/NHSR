import numpy  as np
import HSRautotest
m = np.array([100,200,300,400,500])
n = np.array([200,400,600,800,1000])
lamda = np.array([8])
testrmse = np.eye(5,1)
trainrmse = np.eye(5,1)

for i in range(len(m)):
    for j in range(len(n)):
        for k in range(len(lamda)):
            av = 0
            fo = open("result.txt", "a")
            for q in range(10):
                trainrmse[q],testrmse[q] = HSRautotest.HSR(lamda=lamda[k],m1= m[i],n1 = n[j],n_epochs=100)
                av = av + testrmse[q]

            av = av/10

            str1 = " m1 " + str(m[i]) + " n1 " + str(n[j]) + "lamda " + str(lamda[k])  + " the avg of testrmse is "+ str(av) +"\n"
            fo.write(str1);
            fo.close()
