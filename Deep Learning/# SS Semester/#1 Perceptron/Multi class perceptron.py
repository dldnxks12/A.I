import sys
import numpy as np
import matplotlib.pyplot as plt

X = np.random.rand(100, 2)
Y = np.ones((100,1))

# Define 3 Class (Label : 1, 2, 3)
Y[X[:, 1] <= (3*X[:,0] - 0.5)] = -1
Y[X[:, 1] <= (3*X[:,0] - 1)]   = 0

'''
plt.scatter(X[:, 0], X[:, 1], c=Y, s=10, cmap=plt.cm.Paired)
plt.xlabel('x1')
plt.ylabel('x2')
plt.show()
sys.exit()
'''

### Define perceptron
# Method 1 : one vs rest

# 1번 라벨 : 0 , 2번 라벨 1, 3번 라벨 2
# Method 2 : one vs one
def perceptron(X, Y):
    X_ones = np.concatenate((np.ones((len(X), 1)), X), axis=1) # 100 x 3
    W1 = np.zeros(X_ones.shape[1]) # Class 0 vs Class 1, 2
    W2 = np.zeros(X_ones.shape[1]) # Class 1 vs Class 0, 2
    W3 = np.zeros(X_ones.shape[1]) # Class 2 vs Class 0, 1

    epochs = 100
    for ep in range(epochs):
        Mat_result1 = np.matmul(X_ones, W1)
        Mat_result2 = np.matmul(X_ones, W2)
        Mat_result3 = np.matmul(X_ones, W3)

        for index, values in enumerate(zip(Mat_result1, Mat_result2, Mat_result3)):
            pass

    return W1, W2 # Return trained weights

Weight1, Weight2 = perceptron(X, Y)
print("Weight1", Weight1)
print("Weight2", Weight2)

plt.scatter(X[:, 0], X[:, 1], c=Y, s=10, cmap=plt.cm.Paired)
plt.plot([0,1],[-Weight1[0]/Weight1[2], -Weight1[0]/Weight1[2]-Weight1[1]/Weight1[2]]) # [Xmin, Xmax] , [Ymin, Ymax]
plt.plot([0,1],[-Weight2[0]/Weight2[2], -Weight2[0]/Weight2[2]-Weight2[1]/Weight2[2]]) # [Xmin, Xmax] , [Ymin, Ymax]
plt.xlim((0,1))
plt.ylim((0,1))
plt.xlabel('x1')
plt.ylabel('x2')
plt.show()