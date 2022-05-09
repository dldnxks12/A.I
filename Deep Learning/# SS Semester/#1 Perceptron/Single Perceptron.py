import sys
import numpy as np
import matplotlib.pyplot as plt

### Data
X = np.random.rand(100, 2) # 100 x 2
Y = np.ones((100,1))       # 100 x 1

Y[X[:, 1] <= (3*X[:, 0] - 1)] = -1 # x2 > 3*x1 + 1 -> 1 else -1

assert X.shape == (100,2)
assert Y.shape == (100,1)

'''
### Given Data Visualization
plt.scatter(X[:,0], X[:,1], c = Y, s = 10, cmap = plt.cm.Paired)
plt.xlabel('X1')
plt.xlabel('X2')
plt.show()
'''

### Define perceptron
def perceptron(X, Y):
    X_ones = np.concatenate((np.ones((len(X), 1)), X), axis=1) # 100 x 3
    W = np.zeros(X_ones.shape[1])

    epochs = 100
    for ep in range(epochs):
        Mat_result = np.matmul(X_ones, W)

        for index, value in enumerate(Mat_result):
            if (value * Y[index]) <= 0:
                W = W + (X_ones[index] * Y[index])
    return W # Return trained weights

Weight = perceptron(X, Y)
print("Weight", Weight)
assert Weight.shape[0] == 3

plt.scatter(X[:, 0], X[:, 1], c=Y, s=10, cmap=plt.cm.Paired)
plt.plot([0,1],[-Weight[0]/Weight[2], -Weight[0]/Weight[2]-Weight[1]/Weight[2]]) # [Xmin, Xmax] , [Ymin, Ymax]
plt.xlim((0,1))
plt.ylim((0,1))
plt.xlabel('x1')
plt.ylabel('x2')
plt.show()




