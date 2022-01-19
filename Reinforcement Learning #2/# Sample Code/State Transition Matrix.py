import numpy as np

rho = [0.4, 0.6] # initial State distribution
P = [[0.58, 0.42], [0.66, 0.34]] # state transition matrix

d = rho


for k in range(0, 20):

    d = np.matmul(d, P) # transpose(rho) * P ...

    print(f"Time Stpe : {k+1} Result : {d}")


