import numpy as np

X = np.loadtxt("./data.txt", delimiter= ',')
y = np.loadtxt("./targets.txt") - 1

print(y)

