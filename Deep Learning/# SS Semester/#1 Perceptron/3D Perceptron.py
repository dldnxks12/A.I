import sys
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d

### Data
X = np.random.rand(100, 3) # 100 x 2
Y = np.ones((100,1))       # 100 x 1

Y[X[:, 2] <= (3*X[:, 0] + 2*X[:, 1] - 1)] = -1 # x2 > 3*x1 + 2*x2 - 1 -> 1 else -1

assert X.shape == (100,3)
assert Y.shape == (100,1)

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
assert Weight.shape[0] == 4

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
for m in [('o'), ('x')]:
    ax.scatter(X[:,0], X[:,1], X[:,2], c = Y, s = 10, marker=m)
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

X, Y = np.linspace(0, 1, 11), np.linspace(0,1,11)
Z = (-Weight[0] -Weight[1]*X - Weight[2]*Y) / Weight[3]
ax.plot(X, Y, Z)

# rotate the axes and update
for angle in range(0, 360):
    ax.view_init(30, angle)
    plt.draw()
    plt.pause(.001)


