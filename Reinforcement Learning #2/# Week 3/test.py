import numpy as np


A = np.zeros([2,2])


B = A

B[0,0] = 10

print(A, B)

C = A[:,:]

C[0] = 20

print(C, A)