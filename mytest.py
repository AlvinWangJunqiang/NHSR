import numpy as np
import NMFclassx2
import NMFclass

a = np.array([[1,2,3,4,5],[5,6,7,8,9]])

P1,Q1 = NMFclassx2.NMF(a,5)
P2,Q2 = NMFclass.NMF(a,5)
print a
b = np.dot(P1,Q1)
print b*b
print np.dot(P2,Q2)
