import numpy as np
import NMFclassx2
import NMFclass
import copy
# a = np.array([[1,2,3,4,5],[5,6,7,8,9]])
#
# P1,Q1 = NMFclassx2.NMF(a,5)
# P2,Q2 = NMFclass.NMF(a,5)
# print a
# b = np.dot(P1,Q1)
# print b*b
# print np.dot(P2,Q2)



# a=np.random.random((1,10**7))
# e=np.random.random((10**6,10**5))
#
#
# b=np.random.random((10**7,1))
# import time
#
# start_Real = time.time()
#
# c = np.dot(a,b)
# print c,10**4
# end_End = time.time()
# print("Method 1: %f real seconds" % (end_End - start_Real))

# a = {}
# a[1] = 45
# a [2]  = 46
#
# b= a
# b[1] = b[1] + 3
# print a[1]
#
# c = copy.deepcopy(a)
# print a[1]
# c[1] = a[1] + b[1]
# print c[1],a[1]
#
# class a():
#
#     def __init__(self,m):
#         self.m = m
#
# class d():
#     def __init__(self,w):
#         self.w = w
# class b(a,d):
#     def __init__(self,m,n):
#         self.m = m
#         self.n = n
#         a.__init__(self,m = n)
#         d.__init__(self,w = n)
#
# bb = b(1,5)
#
# print bb.m , bb.n , bb.w


# a = (1,2,3)
# c = str(a)
# print c
# m = 684
# M = [20]
# N = [m]
# N.extend(M[1:])
# N.append(M[0])
# print N
#
# P = 0.01*np.random.rand(100, 100)/100 + 10 ** -9
#
# print np.sum(P)

a= np.array([[12,3,45,6],[1,3,4,6]])
print (a[:,0] - a[:,0].min())/ ( a[:,0].min()- a[:,0].max())